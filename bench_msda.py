#!/usr/bin/env python3
"""Test the deformable-attention CUDA kernel through the real module, not a stub.

The first attempt at this substitution (``fused_msda``) was 3.9-5.0x faster
standalone yet 2.4% slower end to end, and the suspected reason was layout: the
kernel wants ``[batch, sequence, heads, head_dim]`` while
``MSDeformAttn.forward`` hands its core ``[batch, heads, head_dim, sequence]``,
so the patch had to permute and copy the whole value tensor back.

Two things need testing at the module level rather than on synthetic tensors:

* **Correctness of the real call.** Upstream holds ``attention_weights`` as
  ``[batch, queries, heads, levels * points]`` but the kernel's contract is
  ``[batch, queries, heads, levels, points]``. The layouts are identical in
  memory, so the flattened form is expected to work -- but the earlier verifier
  built its own rank-5 inputs and therefore never exercised what the model
  actually passes.
* **Whether removing the copy flips the verdict.** ``value_proj`` already
  produces ``[batch, sequence, channels]``, so the kernel's layout is one free
  ``view`` away; it is upstream's ``transpose(1, 2).contiguous()`` that creates
  the expensive layout. A forward that skips it should beat both.

Every variant is run under the same ``autocast(bfloat16)`` the training loop
uses, with production shapes, and compared against the unpatched module.

    uv run python -m lab.bench_msda
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor

from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

from lab.kernels_msda_cuda import module_forward as kernel_forward_native

#: Production geometry: RF-DETR Medium, resolution 576, 3 feature levels.
DECODER = {"batch": 8, "queries": 300, "shapes": [(36, 36), (18, 18), (9, 9)]}
ENCODER = {"batch": 8, "queries": 1701, "shapes": [(36, 36), (18, 18), (9, 9)]}

D_MODEL = 256
N_HEADS = 8
N_POINTS = 4


#: The module's own core, captured before any patch can replace it. Patching is
#: a global mutation, so every variant restores this afterwards -- without that,
#: the "upstream" rows of later cases silently measure the patched core.
import rfdetr.models.ops.modules.ms_deform_attn as msda_module  # noqa: E402

PRISTINE_CORE = msda_module.ms_deform_attn_core_pytorch


def upstream_forward(module: MSDeformAttn, *args: Any, **kwargs: Any) -> Tensor:
    """The unpatched module forward, with the original core guaranteed."""
    msda_module.ms_deform_attn_core_pytorch = PRISTINE_CORE
    return MSDeformAttn.forward(module, *args, **kwargs)


def kernel_forward_transposed(module: MSDeformAttn, *args: Any, **kwargs: Any) -> Tensor:
    """Upstream forward with only the core swapped -- keeps upstream's copy.

    This is what the rejected ``fused_msda`` patch did: the module still builds
    ``[batch, heads, head_dim, sequence]`` and the replacement permutes it back.
    """
    from lab.patches import apply_patches

    apply_patches("fused_msda")
    try:
        return MSDeformAttn.forward(module, *args, **kwargs)
    finally:
        msda_module.ms_deform_attn_core_pytorch = PRISTINE_CORE


def make_inputs(batch: int, queries: int, shapes: list[tuple[int, int]]) -> dict[str, Any]:
    """Random inputs matching what the decoder/encoder hands the module."""
    generator = torch.Generator(device="cuda").manual_seed(4)
    sequence = sum(height * width for height, width in shapes)
    spatial = torch.tensor(shapes, device="cuda", dtype=torch.int64)
    sizes = spatial[:, 0] * spatial[:, 1]
    return {
        "query": torch.randn((batch, queries, D_MODEL), device="cuda", generator=generator),
        "reference_points": torch.rand((batch, queries, len(shapes), 2), device="cuda", generator=generator),
        "input_flatten": torch.randn((batch, sequence, D_MODEL), device="cuda", generator=generator),
        "input_spatial_shapes": spatial,
        "input_level_start_index": torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1])),
        "input_spatial_shapes_hw": shapes,
    }


def run_variant(
    module: MSDeformAttn, inputs: dict[str, Any], forward: Callable[..., Tensor]
) -> tuple[Tensor, list[Tensor]]:
    """One forward and backward under the training loop's autocast settings."""
    module.zero_grad(set_to_none=True)
    query = inputs["query"].detach().clone().requires_grad_(True)
    flatten = inputs["input_flatten"].detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = forward(module, query=query, reference_points=inputs["reference_points"], input_flatten=flatten,
                         input_spatial_shapes=inputs["input_spatial_shapes"],
                         input_level_start_index=inputs["input_level_start_index"],
                         input_spatial_shapes_hw=inputs["input_spatial_shapes_hw"])
    output.float().pow(2).mean().backward()
    grads = [query.grad.detach().clone(), flatten.grad.detach().clone()]
    grads += [
        parameter.grad.detach().clone()
        for _, parameter in sorted(module.named_parameters())
        if parameter.grad is not None
    ]
    return output.detach().float(), grads


def time_variant(module: MSDeformAttn, inputs: dict[str, Any], forward: Callable[..., Tensor], reps: int = 12) -> float:
    """Median wall time of forward+backward, in milliseconds."""
    for _ in range(3):
        run_variant(module, inputs, forward)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        run_variant(module, inputs, forward)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(0)
    variants = (
        ("upstream (grid_sample)", upstream_forward),
        ("kernel, upstream layout", kernel_forward_transposed),
        ("kernel, native layout", kernel_forward_native),
    )

    for label, geometry in (("decoder", DECODER), ("encoder", ENCODER)):
        module = MSDeformAttn(d_model=D_MODEL, n_levels=len(geometry["shapes"]), n_heads=N_HEADS, n_points=N_POINTS)
        module = module.cuda()
        inputs = make_inputs(**geometry)

        print(f"\n{label}: batch {geometry['batch']}, {geometry['queries']} queries, levels {geometry['shapes']}")
        reference_output, reference_grads = run_variant(module, inputs, upstream_forward)
        print(f"{'variant':26s} {'ms':>8s} {'speedup':>8s}  {'max|d output|':>13s} {'max|d grad|':>12s}")
        baseline_ms = None
        for name, forward in variants:
            output, grads = run_variant(module, inputs, forward)
            output_delta = (output - reference_output).abs().max().item()
            grad_delta = max(
                (actual - expected).abs().max().item() for actual, expected in zip(grads, reference_grads)
            )
            milliseconds = time_variant(module, inputs, forward, args.reps)
            baseline_ms = baseline_ms or milliseconds
            print(
                f"{name:26s} {milliseconds:8.2f} {baseline_ms / milliseconds:7.2f}x  "
                f"{output_delta:13.2e} {grad_delta:12.2e}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
