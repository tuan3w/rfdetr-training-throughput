#!/usr/bin/env python3
"""Route multi-scale deformable attention through the real fused CUDA kernel.

rfdetr 1.10.1 ships only ``ms_deform_attn_core_pytorch``, whose own docstring
says "For debug and test only, need to use cuda version instead", and no compiled
extension is installed (``import MultiScaleDeformableAttention`` fails). So every
training step runs deformable attention as a Python composition: one
``F.grid_sample`` per feature level, a stack, a multiply and a sum.

Nsight shows the price on the backward pass. In a 1.4 s steady window
``grid_sampler_2d_backward_kernel<float, int>`` costs 65.5 ms across 30 calls
(2.18 ms each) -- the largest non-GEMM kernel in the profile -- and it runs in
**fp32** because autocast keeps ``grid_sampler`` out of bf16, so the bf16 value
tensor is upcast first. The fallback also forces an extra
``transpose(1, 2).contiguous()`` copy of the whole value tensor to reach the
layout it wants.

This wraps the Deformable-DETR CUDA kernel (fused forward and hand-written
backward, both of which avoid materialising the per-level sampled tensors) in an
autograd function. The kernel is fetched as a prebuilt binary from the Hugging
Face kernels hub, which publishes a build matching this exact environment
(``torch214-cxx11-cu130-x86_64-linux``) -- the same source `transformers` uses
for its own deformable-DETR models.

    uv run python -m lab.kernels_msda_cuda --verify
    uv run python -m lab.kernels_msda_cuda --bench
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

#: Hub repository and pinned revision of the prebuilt kernel.
KERNEL_REPO = "kernels-community/deformable-detr"
KERNEL_REVISION = "main"

#: Columns processed per kernel launch; the upstream default.
IM2COL_STEP = 64

_kernel: Any = None


def kernel() -> Any:
    """Load (once) the prebuilt deformable-attention kernel."""
    global _kernel
    if _kernel is None:
        from kernels import get_kernel

        _kernel = get_kernel(KERNEL_REPO, revision=KERNEL_REVISION)
    return _kernel


class DeformableAttention(torch.autograd.Function):
    """Fused multi-scale deformable attention with the kernel's own backward."""

    @staticmethod
    def forward(
        ctx: Any,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        ctx.im2col_step = IM2COL_STEP
        output = kernel().ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = kernel().ms_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight



def level_start_index(spatial_shapes: Tensor) -> Tensor:
    """Offset of each level's first element in the flattened value sequence."""
    sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    return torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))


def deformable_attention(
    value_by_head: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Drop-in replacement for ``ms_deform_attn_core_pytorch``.

    Args:
        value_by_head: ``[batch, heads, head_dim, sequence]``, the layout rfdetr's
            fallback expects.
        spatial_shapes: ``[levels, 2]`` of ``(height, width)``.
        sampling_locations: ``[batch, queries, heads, levels, points, 2]`` in
            ``[0, 1]`` coordinates.
        attention_weights: ``[batch, queries, heads, levels, points]``.

    Returns:
        ``[batch, queries, heads * head_dim]``.

    Raises:
        ValueError: If ``sampling_locations`` is the rank-5 export layout, which
            this kernel does not accept.
    """
    if sampling_locations.ndim != 6:
        raise ValueError(f"expected rank-6 sampling locations, got shape {tuple(sampling_locations.shape)}")
    # Run in fp32, which is what the path being replaced already does: autocast
    # keeps `grid_sampler` on its fp32 list, so the fallback upcasts the bf16
    # value tensor and returns fp32. Feeding the kernel bf16 instead would be a
    # real precision change -- bilinear sampling coordinates carry only 8 mantissa
    # bits there, which moves sampling by a fraction of a pixel and shifts the
    # location gradients by ~80% (see verify()). Matching fp32 keeps this patch a
    # pure performance change.
    value = value_by_head.permute(0, 3, 1, 2).to(torch.float32).contiguous()
    shapes = spatial_shapes.to(device=value.device, dtype=torch.int64)
    return DeformableAttention.apply(
        value,
        shapes,
        level_start_index(shapes),
        sampling_locations.to(torch.float32).contiguous(),
        attention_weights.to(torch.float32).contiguous(),
    )


def module_forward(  # noqa: PLR0913
    module: Any,
    query: Tensor,
    reference_points: Tensor,
    input_flatten: Tensor,
    input_spatial_shapes: Tensor,
    input_level_start_index: Tensor,
    input_padding_mask: Tensor | None = None,
    input_spatial_shapes_hw: list[tuple[int, int]] | None = None,
) -> Tensor:
    """The eager forward, reaching the kernel without any layout copy.

    Mirrors ``MSDeformAttn.forward``'s eager branch (rfdetr 1.10.1, lines
    175-259) with two deliberate differences:

    * ``value`` stays ``[batch, sequence, heads, head_dim]``, which is a free
      ``view`` of what ``value_proj`` produced, instead of being transposed and
      copied into ``[batch, heads, head_dim, sequence]``.
    * the fused kernel is called directly, in fp32 -- the precision this path
      already runs at, since autocast keeps ``grid_sampler`` off bf16.

    The export branch and the padding-mask branch are delegated to upstream so
    this only ever handles what training actually exercises.

    Note on why this still does not pay end to end. Nsight traces with and
    without the substitution show the kernel replacing 59.7 ms of
    ``grid_sampler`` with 41.7 ms of ``ms_deformable_im2col``/``col2im`` per 1.2 s
    window, but aten layer-norm kernels around the attention block go from 24 to
    48-54 calls (+16.9 ms) and cast kernels add 12.8 ms: the opaque call splits
    an Inductor-fused region. Registering the kernel as a
    ``torch.library.custom_op`` instead -- the textbook fix for a graph break --
    was worse still (38.67 vs 42.75 img/s), because an opaque node also stops
    Inductor fusing the surrounding casts.
    """
    batch_size, len_query, _ = query.shape
    batch_size, len_input, _ = input_flatten.shape

    value = module.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    # Free view: [batch, sequence, channels] -> [batch, sequence, heads, head_dim].
    value = value.view(batch_size, len_input, module.n_heads, module.d_model // module.n_heads)

    attention_weights = module.attention_weights(query).view(
        batch_size, len_query, module.n_heads, module.n_levels * module.n_points
    )
    sampling_offsets = module.sampling_offsets(query).view(
        batch_size, len_query, module.n_heads, module.n_levels, module.n_points, 2
    )
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets / module.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )
    else:
        raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.")
    attention_weights = F.softmax(attention_weights, -1)

    shapes = input_spatial_shapes.to(device=value.device, dtype=torch.int64)
    output = DeformableAttention.apply(
        value.to(torch.float32).contiguous(),
        shapes,
        level_start_index(shapes),
        sampling_locations.to(torch.float32).contiguous(),
        attention_weights.to(torch.float32).contiguous(),
    )
    return module.output_proj(output)


def _reference(
    value_by_head: Tensor, spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    """rfdetr's own fallback, used as ground truth."""
    from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch

    return ms_deform_attn_core_pytorch(value_by_head, spatial_shapes, sampling_locations, attention_weights)


def _problem(
    batch: int,
    heads: int,
    head_dim: int,
    queries: int,
    points: int,
    shapes: list[tuple[int, int]],
    dtype: torch.dtype,
) -> tuple[Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(7)
    sequence = sum(height * width for height, width in shapes)
    value = torch.randn(
        (batch, heads, head_dim, sequence), device="cuda", dtype=dtype, generator=generator, requires_grad=True
    )
    locations = torch.rand(
        (batch, queries, heads, len(shapes), points, 2), device="cuda", dtype=dtype, generator=generator
    ).requires_grad_(True)
    weights = torch.rand(
        (batch, queries, heads, len(shapes), points), device="cuda", dtype=dtype, generator=generator
    )
    weights = (weights / weights.sum(-1, keepdim=True)).detach().requires_grad_(True)
    spatial = torch.tensor(shapes, device="cuda", dtype=torch.int64)
    return value, spatial, locations, weights


#: Shapes taken from the production model: 3 levels at resolution 576, 8 heads,
#: 32 channels per head, 300 decoder queries and the 4-point sampling pattern.
CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "production decoder",
        {"batch": 8, "heads": 8, "head_dim": 32, "queries": 300, "points": 4, "shapes": [(36, 36), (18, 18), (9, 9)]},
    ),
    (
        "encoder-sized queries",
        {"batch": 4, "heads": 8, "head_dim": 32, "queries": 1296, "points": 4, "shapes": [(36, 36), (18, 18)]},
    ),
    (
        "single level",
        {"batch": 2, "heads": 4, "head_dim": 16, "queries": 64, "points": 4, "shapes": [(20, 20)]},
    ),
)


def verify() -> int:
    """Compare forward and gradients against rfdetr's fallback.

    Only fp32 is checked, because fp32 is the precision the replaced path
    actually runs at under autocast. Running the kernel in bf16 was measured and
    is not equivalent: output differs by ~0.36 against a scale of 3.6 and the
    location gradients by ~530 against a scale of 650, since bf16 sampling
    coordinates cannot address a 36x36 feature map precisely.
    """
    failures = 0
    for label, case in CASES:
        for dtype in (torch.float32,):
            value, spatial, locations, weights = _problem(dtype=dtype, **case)
            reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (value, locations, weights)]

            expected = _reference(reference_inputs[0], spatial, reference_inputs[1], reference_inputs[2])
            actual = deformable_attention(value, spatial, locations, weights)

            seed = torch.randn_like(expected)
            expected.backward(seed)
            actual.backward(seed)

            scale = expected.abs().max().item()
            tolerance = 3e-3 if dtype is torch.bfloat16 else 2e-5
            deltas = {
                "output": (actual - expected).abs().max().item(),
                "grad_value": (value.grad - reference_inputs[0].grad).abs().max().item(),
                "grad_locations": (locations.grad - reference_inputs[1].grad).abs().max().item(),
                "grad_weights": (weights.grad - reference_inputs[2].grad).abs().max().item(),
            }
            # Gradients through bilinear sampling are scaled by the feature-map
            # size, so they are compared against their own magnitude.
            grad_scales = {
                "output": scale,
                "grad_value": reference_inputs[0].grad.abs().max().item(),
                "grad_locations": reference_inputs[1].grad.abs().max().item(),
                "grad_weights": reference_inputs[2].grad.abs().max().item(),
            }
            bad = [
                name
                for name, delta in deltas.items()
                if delta > tolerance * max(1.0, grad_scales[name])
            ]
            status = "ok " if not bad else "FAIL"
            failures += bool(bad)
            summary = "  ".join(f"{name} {delta:.2e}/{grad_scales[name]:.1e}" for name, delta in deltas.items())
            print(f"{status} {label:22s} {str(dtype).replace('torch.', ''):9s} {summary}")
            if bad:
                print(f"     exceeded tolerance {tolerance:.1e} on: {', '.join(bad)}")
    return 1 if failures else 0


def bench() -> int:
    """Time the fused kernel against the fallback, forward and backward."""
    print(f"{'case':24s} {'dtype':9s} {'fallback ms':>12s} {'fused ms':>10s} {'speedup':>8s}")
    for label, case in CASES:
        for dtype in (torch.float32,):
            value, spatial, locations, weights = _problem(dtype=dtype, **case)

            def run(use_kernel: bool) -> float:
                for tensor in (value, locations, weights):
                    tensor.grad = None
                function = deformable_attention if use_kernel else _reference
                output = function(value, spatial, locations, weights)
                seed = torch.ones_like(output)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(5):
                    output = function(value, spatial, locations, weights)
                    output.backward(seed, retain_graph=False)
                torch.cuda.synchronize()
                return (time.perf_counter() - started) / 5 * 1e3

            fallback, fused = run(False), run(True)
            print(
                f"{label:24s} {str(dtype).replace('torch.', ''):9s} {fallback:12.2f} {fused:10.2f} "
                f"{fallback / fused:7.2f}x"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check against rfdetr's fallback.")
    parser.add_argument("--bench", action="store_true", help="Time both implementations.")
    args = parser.parse_args()
    if not (args.verify or args.bench):
        parser.error("choose --verify and/or --bench")
    status = 0
    if args.verify:
        status |= verify()
    if args.bench:
        status |= bench()
    return status


if __name__ == "__main__":
    sys.exit(main())
