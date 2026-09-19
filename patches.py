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


def _apply_cuda_lap() -> str:
    """Solve the assignment on the device instead of shipping the costs to the host.

    RF-DETR reaches ``torch_linear_assignment.batch_linear_assignment``, which on
    this workload lands on the SciPy fallback: it copies every cost matrix to the
    host (Nsight: 11-13 ms/step of pageable device-to-host traffic) and blocks on
    a ``cudaStreamSynchronize`` while the CPU solves. NVTX attributes ~72 ms of
    host time per microbatch to the criterion waiting on that, even with the
    solve parallelised across 8 threads.

    ``lab/kernels_lap.py`` solves the same problems exactly on the GPU — one CUDA
    block per problem, Jonker-Volgenant shortest augmenting paths — in 1.5-1.7 ms
    for the whole step's 520 problems, with nothing leaving the device. Batches
    outside its envelope (non-CUDA, non-float32, ``tasks > workers``, or more
    workers than the shared-memory budget allows) fall through to the original
    implementation, so behaviour is unchanged wherever the kernel declines.
    """
    from torch_linear_assignment import assignment as assignment_module

    from lab.kernels_lap import batch_lap, supported

    original = assignment_module.batch_linear_assignment

    def batch_linear_assignment(cost: Any) -> Any:
        if supported(cost):
            return batch_lap(cost)
        return original(cost)

    assignment_module.batch_linear_assignment = batch_linear_assignment
    # rfdetr imports the function into its own module namespace at call time via
    # `import torch_linear_assignment`, so patching the module attribute is
    # enough; this line keeps the package-level alias consistent for anything
    # that imported it earlier.
    import torch_linear_assignment

    torch_linear_assignment.batch_linear_assignment = batch_linear_assignment
    return "cuda_lap: exact Jonker-Volgenant assignment solved on the GPU (no cost-matrix host transfer)"


def _apply_device_indices() -> str:
    """Unpack the assignment on the device so the criterion never synchronises.

    ``_assignment._assign_padded`` solves the batch, copies the index pairs to the
    host (``_solve_to_indices``'s ``.cpu()``) and drops padded columns there. That
    was the right split when the solve ran on the host, but with ``cuda_lap`` the
    indices are born on the device, and the copy is what makes the criterion
    synchronise: every later ``target["boxes"][target_indices]`` and
    ``outputs["pred_boxes"][idx]`` indexes a CUDA tensor with a CPU index tensor.
    ``lab/sync_probe.py`` attributes 156 synchronising operations per step, ~130 of
    them to exactly those uses (``criterion.py:904``, ``:906``, ``:562``, ``:722``,
    ``:576-580``).

    Two details make the device version sync-free:

    * The padded columns cannot be dropped with a boolean mask, because masked
      indexing has a data-dependent output size and so synchronises itself.
      Instead each problem's pairs are sorted by column: padded columns are
      exactly those at or above the image's target count, so a stable ascending
      sort puts every real column first, and the real pairs are then read with a
      selection index that depends only on ``sizes`` (known on the host already).
    * That selection index is the same for every step with the same target
      counts, so it is cached rather than rebuilt.

    Sorting by column reorders the pairs within an image instead of leaving them
    in row order. The pairing is permuted identically on both sides, and every
    consumer either sums over pairs or scatters into distinct slots, so the loss
    is unchanged -- ``lab/verify_matcher.py`` checks this on real batches.
    """
    import torch

    from rfdetr.models import _assignment

    original = _assignment._assign_padded
    index_cache: dict[tuple, Any] = {}

    def selection_index(sizes: tuple[int, ...], layers: int, group_detr: int, stride: int, device: Any) -> Any:
        """Flat positions of the real (non-padded) pairs, per problem."""
        key = (sizes, layers, group_detr, stride, device)
        cached = index_cache.get(key)
        if cached is None:
            per_problem = [size for _ in range(layers) for size in sizes for _ in range(group_detr)]
            starts = torch.arange(len(per_problem), dtype=torch.int64) * stride
            cached = torch.cat(
                [start + torch.arange(size, dtype=torch.int64) for start, size in zip(starts.tolist(), per_problem)]
            ).to(device, non_blocking=True)
            index_cache[key] = cached
        return cached

    def assign_padded(
        cost_matrices: list[Any], sizes: list[int], group_width: int, group_detr: int
    ) -> list[list[tuple[Any, Any]]]:
        import torch_linear_assignment

        max_size = max(sizes)
        stacked = _assignment._stack_padded(cost_matrices, sizes, group_width, group_detr, max_size)
        assignment = torch_linear_assignment.batch_linear_assignment(stacked)
        if not assignment.is_cuda:
            return original(cost_matrices, sizes, group_width, group_detr)

        num_matches = min(group_width, max_size)
        matched = assignment >= 0
        order = torch.argsort(matched.to(torch.int8), dim=1, descending=True, stable=True)
        rows = order[:, :num_matches]
        cols = assignment.gather(1, rows)

        # Real columns first, padded ones last, without a data-dependent mask.
        by_column = torch.argsort(cols, dim=1, stable=True)
        group_offset = torch.arange(rows.shape[0], device=rows.device, dtype=rows.dtype) % group_detr * group_width
        kept_rows = (rows + group_offset.unsqueeze(1)).gather(1, by_column).reshape(-1)
        kept_cols = cols.gather(1, by_column).reshape(-1)

        flat = selection_index(tuple(sizes), len(cost_matrices), group_detr, num_matches, rows.device)
        kept_rows = kept_rows.index_select(0, flat)
        kept_cols = kept_cols.index_select(0, flat)

        per_image = [group_detr * size for _ in cost_matrices for size in sizes]
        row_chunks = torch.split(kept_rows, per_image)
        col_chunks = torch.split(kept_cols, per_image)
        return [
            [
                (
                    row_chunks[layer_index * len(sizes) + image_index],
                    col_chunks[layer_index * len(sizes) + image_index],
                )
                for image_index in range(len(sizes))
            ]
            for layer_index in range(len(cost_matrices))
        ]

    _assignment._assign_padded = assign_padded
    return "device_indices: assignment unpacked on the GPU (no host round trip in the criterion)"


def _apply_compile_backbone() -> str:
    """Stop one tiny op from forcing the whole DINOv2 backbone into eager mode.

    With ``compile=True`` every graph that reaches the backbone fails to compile:

        BackendCompilerFailed: backend='inductor' raised:
        RuntimeError: isIntList() INTERNAL ASSERT FAILED ... Expected IntList but
        got GenericList

    ``tlparse`` puts the blame precisely (7 of the traced compile ids, frames 0-5
    and 7). The failing node is the positional-embedding resize in
    ``dinov2_with_windowed_attn.interpolate_pos_encoding``::

        _upsample_bicubic2d_aa_backward(grad, [floordiv, floordiv],
                                        [1, 384, 36, 36], False)

    ``rfdetr`` compiles with ``dynamic=True`` so that one graph serves all
    multi-scale resolutions, which makes the resize target the symbolic
    ``(s97 // 16)``; the antialiased-bicubic *backward* cannot take a symbolic
    output size, so the whole graph is rejected. ``suppress_errors=True`` then
    hides it and the backbone -- the dominant compute in the model -- silently
    runs eager. Upstream knows: ``module_model.py:439`` names this exact
    subgraph as the reason for keeping ``suppress_errors`` on.

    The fix keeps ``dynamic=True`` and evicts only the offending op: disabling
    Dynamo for ``interpolate_pos_encoding`` graph-breaks around one bicubic
    resize of a ``[1, 384, 36, 36]`` tensor and lets inductor compile everything
    around it. Specialising the resolution instead would compile the resize too,
    but at the cost of a separate graph per multi-scale size.
    """
    import torch

    from rfdetr.models.backbone import dinov2_with_windowed_attn as windowed

    embeddings = windowed.WindowedDinov2WithRegistersEmbeddings
    embeddings.interpolate_pos_encoding = torch.compiler.disable(  # type: ignore[method-assign]
        embeddings.interpolate_pos_encoding
    )
    return "compile_backbone: pos-embed resize left eager so inductor can compile the backbone"


def _apply_static_compile() -> str:
    """Compile with concrete shapes instead of symbolic ones.

    ``module_model.py:450`` compiles with ``dynamic=True`` so that one graph
    serves every multi-scale resolution. The cost of that choice is severe:
    the model's FX graph is ~800 KB of nodes whose shapes are all expressions in
    one symbol, and inductor's shape reasoning over it is single-threaded sympy
    work -- a single frame ran for 22 minutes without finishing, with the GPU
    idle and one core pinned. ``suppress_errors=True`` hides the fallout, so the
    model forward silently runs eager (``tlparse`` shows compile ids 0-5 and 7,
    all rooted at ``lwdetr.py:477``, failing with ``BackendCompilerFailed``).

    With ``dynamic=False`` each resolution gets its own graph with concrete
    shapes, which removes the symbolic reasoning entirely. The multi-scale
    sampler draws from a fixed, small set of sizes, so the number of graphs is
    bounded -- the recompile limits are raised to fit them, and inductor's
    on-disk FX graph cache means a production run pays the compile once.

    ``capture_scalar_outputs`` is turned back off: upstream enables it only
    because ``dynamic=True`` makes ``.item()`` results backed symbols. Without
    dynamic they are unbacked, so the safe behaviour is to let those sites graph
    break instead.
    """
    import torch

    original = torch.compile

    def compile_static(model: Any = None, **kwargs: Any) -> Any:
        kwargs["dynamic"] = False
        return original(model, **kwargs)

    torch.compile = compile_static  # type: ignore[assignment]
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    torch._dynamo.config.capture_scalar_outputs = False
    return "static_compile: dynamic=False so inductor sees concrete shapes"


def _apply_compile_blocks() -> str:
    """Compile the backbone's transformer blocks individually.

    The model as a whole never compiles: the positional-embedding resize in
    ``interpolate_pos_encoding`` rejects the graph (see REJECTED in
    lab/baseline.py), so ``suppress_errors=True`` runs the forward eagerly. Nsight
    shows what that costs: with the device 85.4% busy, 48% of the remaining idle
    is 10,612 gaps of 5-50 us -- launch latency from the many small unfused
    elementwise kernels an eager transformer block emits.

    Compiling the whole graph to fix that was measured and rejected (22 minutes on
    one frame with ``dynamic=True``, 2-3 min per resolution with ``dynamic=False``
    across 8 multi-scale sizes). This takes the opposite approach: compile each
    ``WindowedDinov2WithRegistersLayer`` on its own. The blocks are 16 instances of
    one small module, and torch 2.14 inlines module parameters as graph inputs, so
    they share a single compiled graph -- a small graph that compiles quickly,
    while the embeddings module with the offending resize is never traced at all.

    Left deliberately alone: the patch-embedding and pos-embed path (the thing
    that cannot compile) and the decoder, so this measures exactly one change.
    """
    import torch

    from rfdetr.models.backbone import dinov2_with_windowed_attn as windowed

    layer = windowed.WindowedDinov2WithRegistersLayer
    if getattr(layer, "_lab_compiled", False):
        return "compile_blocks: already applied"
    layer.forward = torch.compile(layer.forward, dynamic=True)  # type: ignore[method-assign]
    layer._lab_compiled = True  # type: ignore[attr-defined]
    return "compile_blocks: DINOv2 blocks compiled individually (one shared graph)"


def _msda_original() -> Any:
    """The unpatched deformable-attention core, captured before replacement."""
    from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch

    return ms_deform_attn_core_pytorch


_ORIGINAL_MSDA_CORE = None


def _apply_fused_msda() -> str:
    """Replace the deformable-attention fallback with the real fused CUDA kernel.

    rfdetr 1.10.1 has no compiled deformable-attention extension
    (``import MultiScaleDeformableAttention`` fails), so it runs
    ``ms_deform_attn_core_pytorch`` -- a function whose own docstring says "For
    debug and test only, need to use cuda version instead". It composes one
    ``F.grid_sample`` per feature level plus a stack, multiply and sum.

    Nsight shows it is the largest non-GEMM cost in the profile: in a 1.4 s
    steady window ``grid_sampler_2d_backward_kernel<float, int>`` takes 65.5 ms
    over 30 calls, in fp32 because autocast keeps ``grid_sampler`` off bf16.

    ``lab/kernels_msda_cuda.py`` wraps the Deformable-DETR kernel (fused forward,
    hand-written backward) fetched prebuilt from the Hugging Face kernels hub for
    this exact torch/CUDA build. It runs in fp32, matching what autocast already
    does to this path, so the substitution is numerically equivalent rather than a
    precision trade: forward matches to 3.2e-06 and gradients to 7.5e-04 against
    location gradients of scale 5.7e+02. Standalone it is 3.9-5.0x faster on the
    production shapes, forward and backward together.
    """
    from rfdetr.models.ops.modules import ms_deform_attn as module

    from lab.kernels_msda_cuda import deformable_attention, kernel

    global _ORIGINAL_MSDA_CORE
    if _ORIGINAL_MSDA_CORE is None:
        _ORIGINAL_MSDA_CORE = _msda_original()

    kernel()  # surface a download or ABI problem here rather than mid-step

    def core(
        value: Any,
        value_spatial_shapes: Any,
        sampling_locations: Any,
        attention_weights: Any,
        value_spatial_shapes_hw: Any = None,
    ) -> Any:
        # The export path passes rank-5 sampling locations, which the kernel does
        # not accept; training never takes it, but keep the fallback reachable.
        if sampling_locations.ndim != 6 or not value.is_cuda:
            return _ORIGINAL_MSDA_CORE(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
                value_spatial_shapes_hw=value_spatial_shapes_hw,
            )
        return deformable_attention(value, value_spatial_shapes, sampling_locations, attention_weights)

    module.ms_deform_attn_core_pytorch = core
    return "fused_msda: deformable attention uses the fused CUDA kernel (3.9-5.0x standalone)"


def _apply_fused_msda_native() -> str:
    """Reach the fused deformable-attention kernel without any layout copy.

    ``fused_msda`` replaced only the core, so the module still built
    ``[batch, heads, head_dim, sequence]`` and the kernel had to permute it back;
    it was 2.4% slower end to end despite being faster standalone. Measured at
    the module level (``lab/bench_msda.py``, autocast bf16, production geometry),
    the copy is a real cost:

        decoder  300 queries: upstream 2.75 ms  ->  1.18 ms copied  ->  1.03 ms native
        encoder 1701 queries: upstream 12.69 ms ->  3.74 ms copied  ->  3.61 ms native

    This replaces ``MSDeformAttn.forward`` itself, so ``value_proj``'s output
    reaches the kernel through a free ``view``. Output matches upstream to
    3.9e-03 -- the bf16 quantum of the final projection -- and gradients to
    4.8e-07.

    The replacement handles only the eager branch; export mode and any call with
    rank-5 sampling locations fall back to the upstream forward. It duplicates
    upstream's sampling-location arithmetic, so it is pinned to rfdetr 1.10.1 and
    is checked against the real module by ``lab/bench_msda.py``.
    """
    from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

    from lab.kernels_msda_cuda import kernel, module_forward

    kernel()  # surface a download or ABI problem here rather than mid-step
    original = MSDeformAttn.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_export", False):
            return original(self, *args, **kwargs)
        return module_forward(self, *args, **kwargs)

    MSDeformAttn.forward = forward  # type: ignore[method-assign]
    return "fused_msda_native: deformable attention kernel reached without a layout copy"


def _apply_bicubic_meta() -> str:
    """Supply the missing meta kernel so the model's graph can compile at all.

    ``aten._upsample_bicubic2d_aa_backward`` is absent from the ``register_meta``
    list in ``torch/_meta_registrations.py`` that already covers
    ``_upsample_bilinear2d_aa_backward`` and ``_upsample_lanczos2d_aa_backward``,
    so with a symbolic output size it reaches the C++ structured kernel and
    ``isIntList()`` fires an internal assert. That is what rejects every graph
    containing the DINOv2 backbone under ``dynamic=True``
    (https://github.com/pytorch/pytorch/issues/197622).

    ``register_meta`` only fills a table; the wiring happens once at import in
    ``activate_meta()`` via ``op.py_impl(DispatchKey.Meta)``. So the fix has to be
    applied the same way -- registering into the table afterwards does nothing,
    which is why an earlier attempt looked like it had failed.

    Verified locally with the same meta function bilinear and lanczos share:
    the compiled dynamic backward matches eager bit-exactly on the output and to
    6e-07 on the gradient across three output sizes.
    """
    import torch
    from torch import _meta_registrations as registrations
    from torch._meta_registrations import meta_upsample_bimode2d_aa_backward

    operator = torch.ops.aten._upsample_bicubic2d_aa_backward.default
    # Testing the Meta dispatch key is useless here: the op already has the C++
    # structured kernel, and that is precisely the one that asserts. What is
    # missing is the *Python* meta, so check the registry `activate_meta()` reads.
    if operator in getattr(registrations, "meta_table", {}):
        return "bicubic_meta: skipped, this torch already registers the python meta"
    operator.py_impl(torch._C.DispatchKey.Meta)(meta_upsample_bimode2d_aa_backward)
    return "bicubic_meta: registered the missing _upsample_bicubic2d_aa_backward meta"


def _sdpa_only(flash: bool, mem_efficient: bool, cudnn: bool, label: str) -> str:
    """Restrict SDPA to one backend so the choice can be measured, not assumed.

    ``sdpa=auto`` enables flash, mem-efficient and cuDNN and lets torch pick;
    it picks flash, and flash's backward is ~19 ms/step here (1271 us per call,
    the second largest kernel after the GEMMs). Whether that is the best
    available on this card is a question nobody has asked.
    """
    import torch

    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient)
    torch.backends.cuda.enable_cudnn_sdp(cudnn)
    torch.backends.cuda.enable_math_sdp(True)  # keep the fallback reachable
    return f"sdpa_{label}: flash={flash} mem_efficient={mem_efficient} cudnn={cudnn}"


def _apply_sdpa_cudnn() -> str:
    """Use cuDNN's fused attention instead of flash."""
    return _sdpa_only(flash=False, mem_efficient=False, cudnn=True, label="cudnn")


def _apply_sdpa_mem_efficient() -> str:
    """Use the mem-efficient (xformers-style) kernel instead of flash."""
    return _sdpa_only(flash=False, mem_efficient=True, cudnn=False, label="mem_efficient")


def _apply_sdpa_flash() -> str:
    """Pin flash explicitly, as the control for the other two."""
    return _sdpa_only(flash=True, mem_efficient=False, cudnn=False, label="flash")


PATCHES = {
    "cdist_l1": _apply_cdist_l1,
    "pinned_d2h": _apply_pinned_d2h,
    "triton_matcher": _apply_triton_matcher,
    "compiled_criterion": _apply_compiled_criterion,
    "compiled_criterion_forward": _apply_compiled_criterion_forward,
    "nvtx": _apply_nvtx,
    "parallel_lap": _apply_parallel_lap,
    "cuda_lap": _apply_cuda_lap,
    "device_indices": _apply_device_indices,
    "bicubic_meta": _apply_bicubic_meta,
    "sdpa_cudnn": _apply_sdpa_cudnn,
    "sdpa_mem_efficient": _apply_sdpa_mem_efficient,
    "sdpa_flash": _apply_sdpa_flash,
    "compile_backbone": _apply_compile_backbone,
    "static_compile": _apply_static_compile,
    "compile_blocks": _apply_compile_blocks,
    "fused_msda": _apply_fused_msda,
    "fused_msda_native": _apply_fused_msda_native,
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
