#!/usr/bin/env python3
"""Kernel-level patches for hot paths the profiler identified.

Each patch is opt-in (``--patch name,name``), applied by monkeypatching so the
vendored rf-detr checkout stays untouched, and must be numerically equivalent to
what it replaces — ``python -m lab.patches --verify`` checks that before any
throughput claim is made.

Available patches
-----------------

``cdist_l1``
    ``HungarianMatcher`` builds its L1 box cost with ``torch.cdist(pred, tgt,
    p=1)`` (``rfdetr/models/matcher.py:445`` and ``:765``) on tensors whose last
    dimension is 4. On CUDA that dispatches to the generic
    ``cdist_kernel_cuda_impl``, which the profiler showed costing **37.6 ms/step
    in 5 calls — 15.3% of all device time in the optimised config**, the single
    most expensive operator. For a 4-wide feature dimension a broadcast
    ``(a - b).abs().sum(-1)`` is a plain elementwise+reduction pair that runs at
    memory bandwidth instead. The patch routes only ``p == 1`` calls with a small
    trailing dimension through the broadcast form and leaves every other call on
    the original implementation.

``pinned_d2h``
    The SciPy assignment path copies each cost matrix to the host with a plain
    ``.cpu()`` (``rfdetr/models/_assignment.py:305``), which lands in *pageable*
    memory — the profiler shows it as ``Memcpy DtoH (Device -> Pageable)``,
    10.0 ms/step, fully synchronous. Staging through a cached pinned buffer lets
    the driver DMA instead of falling back to a staged pageable copy.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

#: Largest trailing dimension routed through the broadcast L1 form. Boxes are 4
#: wide; beyond a handful of columns the materialised difference tensor stops
#: being cheaper than the dedicated kernel.
MAX_BROADCAST_FEATURES = 8


def _apply_cdist_l1() -> str:
    import torch

    original_cdist = torch.cdist

    def cdist(x1: Any, x2: Any, p: float = 2.0, *args: Any, **kwargs: Any) -> Any:
        if (
            p == 1
            and x1.is_cuda
            and x1.dim() >= 2
            and x1.shape[-1] <= MAX_BROADCAST_FEATURES
            and x1.shape[-1] == x2.shape[-1]
        ):
            return (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs().sum(-1)
        return original_cdist(x1, x2, p, *args, **kwargs)

    torch.cdist = cdist
    return "cdist_l1: torch.cdist(p=1, features<=8, cuda) -> broadcast |a-b|.sum(-1)"


def _apply_pinned_d2h() -> str:
    import torch

    from rfdetr.models import _assignment

    # One buffer per (list position, dtype, size): the solver consumes the whole
    # list within a single call, so distinct positions must not share a buffer,
    # while successive calls may safely reuse the same one. Keeping a dedicated
    # buffer per position is what lets the copy skip a defensive clone — the
    # earlier version cloned every matrix and gave the gain straight back.
    buffers: dict[tuple[int, torch.dtype, int], torch.Tensor] = {}

    def to_host(position: int, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a device tensor to the host through a cached pinned staging buffer."""
        if not tensor.is_cuda:
            return tensor
        key = (position, tensor.dtype, tensor.numel())
        staging = buffers.get(key)
        if staging is None:
            staging = torch.empty(tensor.numel(), dtype=tensor.dtype, pin_memory=True)
            buffers[key] = staging
        staging.copy_(tensor.reshape(-1), non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return staging.view(tensor.shape)

    original_solve = _assignment.assign_many_bucketed

    def assign_many_bucketed(cost_matrices: list[torch.Tensor], *args: Any, **kwargs: Any) -> Any:
        staged = [to_host(position, matrix) for position, matrix in enumerate(cost_matrices)]
        return original_solve(staged, *args, **kwargs)

    _assignment.assign_many_bucketed = assign_many_bucketed
    return "pinned_d2h: cost-matrix host transfers staged through pinned memory"


def _apply_triton_matcher() -> str:
    """Replace the matcher's cost-matrix construction with the fused Triton kernel.

    Substitutes ``HungarianMatcher._compute_compact_detection_cost_matrix``, the
    only cost path this workload takes (verified at the call site: 5 calls per
    training step, all compact, all under ``no_grad``). The kernel folds the class
    gather, focal cost, L1 ``cdist``, box conversion, GIoU and the weighted
    combine into one launch; see ``lab/kernels.py`` for the derivation and
    ``lab/verify_matcher.py`` for the real-data equivalence check.

    The padded-to-compact slicing stays in Python because it is pure view work:
    each image contributes only its real target columns, which the kernel's
    padded output already holds.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from rfdetr.models.matcher import HungarianMatcher

    from lab.kernels import fused_matcher_cost

    upstream = HungarianMatcher._compute_compact_detection_cost_matrix

    def compute_compact_detection_cost_matrix(
        self: Any,
        outputs: dict[str, Any],
        targets: list[Any],
        *,
        clamp_target_labels: bool = False,
    ) -> Any:
        if "pred_masks" in outputs or not outputs["pred_boxes"].is_cuda:
            return upstream(self, outputs, targets, clamp_target_labels=clamp_target_labels)

        sizes = [len(target["boxes"]) for target in targets]
        if not any(sizes):
            return upstream(self, outputs, targets, clamp_target_labels=clamp_target_labels)

        padded_labels = pad_sequence([target["labels"] for target in targets], batch_first=True)
        padded_boxes = pad_sequence([target["boxes"] for target in targets], batch_first=True)
        padded_cost = fused_matcher_cost(
            outputs["pred_boxes"],
            padded_boxes,
            outputs["pred_logits"],
            padded_labels,
            cost_bbox=self.cost_bbox,
            cost_class=self.cost_class,
            cost_giou=self.cost_giou,
            focal_alpha=self.focal_alpha,
        )
        return torch.cat([padded_cost[index, :, :size] for index, size in enumerate(sizes)], dim=-1)

    HungarianMatcher._compute_compact_detection_cost_matrix = compute_compact_detection_cost_matrix
    return "triton_matcher: fused cost matrix (gather+focal+L1+GIoU+combine) in one Triton kernel"


def _apply_nvtx() -> str:
    """Install NVTX phase ranges (see ``lab/nvtx.py``); host-side markers only."""
    from lab.nvtx import apply_nvtx_ranges

    return apply_nvtx_ranges()


def _apply_compiled_criterion() -> str:
    """Compile the detection criterion's per-loss functions.

    ``rfdetr`` compiles the model forward but leaves ``SetCriterion`` eager, and
    the criterion runs its loss set once for the final layer plus every auxiliary
    decoder layer plus the encoder layer — five times per microbatch. The
    profiler shows the step is host-bound (``gpu_busy`` 0.46 with ~5,000 kernel
    launches), and those loss chains are a large share of the launch count.

    ``loss_labels`` and ``loss_boxes`` are compiled rather than ``forward``,
    because ``forward`` calls the matcher, whose SciPy solve forces a host sync
    that would graph-break every trace. ``dynamic=True`` is required: the target
    count changes with every batch.
    """
    import torch

    from rfdetr.models.criterion import SetCriterion

    for name in ("loss_labels", "loss_boxes"):
        original = getattr(SetCriterion, name)
        setattr(SetCriterion, name, torch.compile(original, dynamic=True))
    return "compiled_criterion: torch.compile(dynamic=True) on SetCriterion.loss_labels/loss_boxes"


def _apply_parallel_lap() -> str:
    """Solve the batched LAP sub-problems on a thread pool instead of serially.

    NVTX attribution puts ~102 ms of host time per microbatch inside
    ``criterion``, most of it blocked on the SciPy solve that follows the
    cost-matrix transfer: the GPU has nothing queued while the host works through
    the sub-problems one at a time. A step presents ~520 of them (5 layers x 8
    images x 13 query groups), measured at 22.1 ms serially versus 13.5 ms on 8
    threads, because SciPy's C++ solver releases the GIL for part of its work.

    Replaces ``torch_linear_assignment.batch_linear_assignment_cpu`` keeping its
    contract exactly: same dtype preparation, same ``-1``-filled matching tensor,
    same scatter of column indices onto row indices.
    """
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from scipy.optimize import linear_sum_assignment
    from torch_linear_assignment import assignment as assignment_module

    #: Below this many sub-problems the pool hand-off costs more than it saves.
    minimum_batch = 8
    #: Thread count, overridable with ``LAP_THREADS`` for sweeps. The default
    #: leaves the box's remaining cores to the 16 DataLoader workers.
    threads = int(os.environ.get("LAP_THREADS", "8"))
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="lap")

    def batch_linear_assignment_cpu(cost: torch.Tensor) -> torch.Tensor:
        cost = assignment_module._prepare_solver_cost(cost)
        batch_size, workers, _ = cost.shape
        matching = torch.full([batch_size, workers], -1, dtype=torch.long, device=cost.device)
        arrays = cost.numpy()
        solved = (
            map(linear_sum_assignment, arrays)
            if batch_size < minimum_batch
            else pool.map(linear_sum_assignment, arrays)
        )
        for batch_index, (row_indices, column_indices) in enumerate(solved):
            matching[batch_index].scatter_(
                0,
                torch.from_numpy(row_indices),
                torch.from_numpy(column_indices),
            )
        return matching

    assignment_module.batch_linear_assignment_cpu = batch_linear_assignment_cpu
    return f"parallel_lap: SciPy assignment sub-problems solved on a {threads}-thread pool"


def _apply_stacked_costs() -> str:
    """Let the matcher build every layer's cost matrix in one pass.

    Upstream folds all layers into a single cost-construction call only when the
    padded element count stays under ``_STACKED_COST_ELEMENT_LIMIT`` (350k),
    calibrated for the PyTorch op chain. This workload sits far above it (3900
    queries x ~200 padded targets x 5 layers), so it builds each layer
    separately. With the fused Triton kernel the whole stack costs ~0.15 ms, so
    the budget no longer describes reality; raising it lets the matcher issue one
    kernel and one host transfer instead of five.
    """
    from rfdetr.models import matcher as matcher_module

    matcher_module._STACKED_COST_ELEMENT_LIMIT = 1 << 40
    return "stacked_costs: matcher folds all layers into one cost-construction pass"


def _apply_compiled_criterion_forward() -> str:
    """Compile the whole criterion, graph breaks included.

    ``compiled_criterion`` compiled only ``loss_labels``/``loss_boxes`` and moved
    nothing. The Nsight trace says the step issues ~1,900 elementwise kernels for
    7.6 ms of work, so the cost is launch count rather than kernel time, and most
    of those launches come from ``SetCriterion.forward``: it runs its loss set for
    the final layer, each auxiliary decoder layer and the encoder layer, with the
    Python dict/loop plumbing in between.

    Compiling ``forward`` means Dynamo traces across that plumbing and fuses the
    elementwise regions, breaking the graph at the matcher's host sync rather than
    refusing to compile. ``dynamic=True`` is mandatory: the target count changes
    every batch.
    """
    import torch

    from rfdetr.models.criterion import SetCriterion

    SetCriterion.forward = torch.compile(SetCriterion.forward, dynamic=True)
    return "compiled_criterion_forward: torch.compile(dynamic=True) on SetCriterion.forward"


def _apply_triton_msda() -> str:
    """Route deformable attention through the fused Triton kernel.

    Patches the name in ``rfdetr.models.ops.modules.ms_deform_attn``, which is
    where the module imported it, so the decoder picks it up. Multi-level and
    non-CUDA calls fall back to upstream inside the wrapper, so segmentation and
    export paths are untouched. See ``lab/kernels_msda.py``.
    """
    from rfdetr.models.ops.modules import ms_deform_attn as module

    from lab.kernels_msda import fused_ms_deform_attn

    module.ms_deform_attn_core_pytorch = fused_ms_deform_attn
    return "triton_msda: fused deformable attention (gather+bilinear+weight+sum) with fused backward"


PATCHES = {
    "cdist_l1": _apply_cdist_l1,
    "pinned_d2h": _apply_pinned_d2h,
    "triton_matcher": _apply_triton_matcher,
    "compiled_criterion": _apply_compiled_criterion,
    "compiled_criterion_forward": _apply_compiled_criterion_forward,
    "nvtx": _apply_nvtx,
    "parallel_lap": _apply_parallel_lap,
    "triton_msda": _apply_triton_msda,
    "stacked_costs": _apply_stacked_costs,
}


def apply_patches(names: str) -> list[str]:
    """Apply a comma-separated patch list; returns one description per patch."""
    applied: list[str] = []
    for name in [item.strip() for item in names.split(",") if item.strip()]:
        if name not in PATCHES:
            raise KeyError(f"unknown patch {name!r}; available: {sorted(PATCHES)}")
        applied.append(PATCHES[name]())
    return applied


def verify() -> int:
    """Check every patch against the implementation it replaces."""
    import torch

    if not torch.cuda.is_available():
        print("verify needs CUDA", file=sys.stderr)
        return 2

    reference = torch.cdist
    failures = 0

    # cdist_l1: exercise the shapes the matcher actually produces — stacked
    # layers x batch, 300 queries x group_detr 13, and a few target counts.
    cases = [(40, 3900, 4, 30), (8, 300, 4, 1), (5, 3900, 4, 97), (3, 100, 4, 0)]
    _apply_cdist_l1()
    patched = torch.cdist
    for batch, queries, features, targets in cases:
        pred = torch.rand(batch, queries, features, device="cuda")
        tgt = torch.rand(batch, targets, features, device="cuda")
        expected = reference(pred, tgt, p=1)
        actual = patched(pred, tgt, p=1)
        if expected.shape != actual.shape:
            print(f"FAIL cdist_l1 shape {expected.shape} != {actual.shape}", file=sys.stderr)
            failures += 1
            continue
        gap = (expected - actual).abs().max().item() if expected.numel() else 0.0
        scale = expected.abs().max().item() if expected.numel() else 1.0
        tolerance = 1e-5 * max(scale, 1.0)
        status = "ok" if gap <= tolerance else "FAIL"
        failures += status == "FAIL"
        print(f"{status} cdist_l1 b{batch} q{queries} t{targets}: max abs diff {gap:.3e} (tol {tolerance:.1e})")

    # p != 2 fallback must still reach the original kernel, and non-L1 metrics
    # must be untouched.
    pred = torch.rand(4, 64, 4, device="cuda")
    tgt = torch.rand(4, 7, 4, device="cuda")
    for order in (2.0, 3.0):
        gap = (reference(pred, tgt, p=order) - patched(pred, tgt, p=order)).abs().max().item()
        status = "ok" if gap == 0.0 else "FAIL"
        failures += status == "FAIL"
        print(f"{status} cdist_l1 fallback p={order}: max abs diff {gap:.3e}")
    torch.cdist = reference

    # pinned_d2h: the staged copy must reproduce `.cpu()` exactly, including when
    # a later call of a different shape reuses the staging buffer.
    from rfdetr.models import _assignment

    original_solve = _assignment.assign_many_bucketed
    staged: list[torch.Tensor] = []

    def capture(cost_matrices: list[torch.Tensor], *args: Any, **kwargs: Any) -> list[Any]:
        staged.extend(cost_matrices)
        return []

    _assignment.assign_many_bucketed = capture
    _apply_pinned_d2h()  # wraps `capture`, so the staged tensors are observable
    try:
        for shape in ((3900, 128), (3900, 64), (300, 7), (3900, 128)):
            matrix = torch.rand(shape, device="cuda")
            staged.clear()
            _assignment.assign_many_bucketed([matrix])
            copied = staged[0]
            identical = torch.equal(copied, matrix.cpu())
            status = "ok" if identical and not copied.is_cuda else "FAIL"
            failures += status == "FAIL"
            print(f"{status} pinned_d2h {shape}: identical to .cpu() = {identical}, on_host = {not copied.is_cuda}")
    finally:
        _assignment.assign_many_bucketed = original_solve

    # parallel_lap: the pooled solver must return exactly what the serial loop
    # returns, for batch sizes above and below the pool threshold.
    from torch_linear_assignment import assignment as assignment_module

    serial_solver = assignment_module.batch_linear_assignment_cpu
    _apply_parallel_lap()
    pooled_solver = assignment_module.batch_linear_assignment_cpu
    try:
        for batch, workers, targets in ((520, 300, 37), (40, 3900, 12), (4, 300, 5), (1, 64, 3)):
            cost = torch.rand(batch, workers, targets)
            identical = torch.equal(serial_solver(cost), pooled_solver(cost))
            status = "ok" if identical else "FAIL"
            failures += status == "FAIL"
            print(f"{status} parallel_lap b{batch} w{workers} t{targets}: identical to serial = {identical}")
    finally:
        assignment_module.batch_linear_assignment_cpu = serial_solver

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check numerical equivalence and exit.")
    parser.add_argument("--list", action="store_true", help="List available patches.")
    args = parser.parse_args()
    if args.list:
        for name in sorted(PATCHES):
            print(name)
        return 0
    if args.verify:
        return verify()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
