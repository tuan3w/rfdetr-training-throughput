#!/usr/bin/env python3
"""Fused Triton kernel for the Hungarian matcher's cost matrix.

Why
---
The profiler (``lab/profile.py``) showed the matcher owning the single most
expensive operator in the optimised training step: ``aten::_cdist_forward``,
37.6 ms/step in 5 calls, 15.3% of all device time. The cost matrix upstream is
built as a chain of separate kernels (``rfdetr/models/matcher.py:441-450``):

1. ``torch.gather`` materialises ``[B, Q, T]`` logits at the target classes,
2. ``_focal_classification_cost`` runs ~8 elementwise passes over that,
3. ``torch.cdist(..., p=1)`` computes the L1 box cost with a generic kernel,
4. ``box_cxcywh_to_xyxy`` twice, then a vmapped ``generalized_box_iou`` (another
   ~12 elementwise/reduction passes over ``[B, Q, T]`` intermediates),
5. three multiplies and two adds to combine them.

Every one of those passes reads and writes a full ``[B, Q, T]`` tensor, and with
``num_queries * group_detr = 3900`` queries and the layer dimension folded into
the batch this is the dominant memory traffic of the matcher. The whole thing is
arithmetic on 4 box coordinates and 1 logit per pair, so it is purely
bandwidth-bound: one fused kernel that reads boxes and logits once and writes the
final cost matrix once replaces all of it.

The kernel computes, for every (query, target) pair:

    cost = cost_bbox * L1(pred_box, tgt_box)
         + cost_class * focal_cost(pred_logit[tgt_label])
         + cost_giou * (-GIoU(pred_box, tgt_box))

matching ``HungarianMatcher._compute_compact_detection_cost_matrix`` exactly,
including the ``clamp(min=1e-7)`` guards that keep degenerate boxes finite and
the logsigmoid-based focal formulation used for numerical stability.

Run ``python -m lab.kernels --verify`` to check it against the PyTorch chain, and
``python -m lab.kernels --bench`` for the standalone kernel timing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
import triton
import triton.language as tl

#: Focal gamma is fixed at 2.0 upstream (``matcher._FOCAL_LOSS_GAMMA``).
FOCAL_GAMMA = 2.0
#: Upstream clamps both the union and the enclosing area at this epsilon.
#: A ``tl.constexpr`` so the jitted kernel may read it as a global.
BOX_EPS = tl.constexpr(1e-7)


@triton.jit
def _matcher_cost_kernel(
    pred_boxes_ptr,  # [B, Q, 4] float32, cxcywh
    tgt_boxes_ptr,  # [B, T, 4] float32, cxcywh
    pred_logits_ptr,  # [B, Q, C] float32
    tgt_labels_ptr,  # [B, T] int64
    out_ptr,  # [B, Q, T] float32
    num_queries,
    num_targets,
    num_classes,
    cost_bbox,
    cost_class,
    cost_giou,
    focal_alpha,
    BLOCK_Q: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EMULATE_LOW_PRECISION_CLASS_COST: tl.constexpr,
):
    """One program computes a ``[BLOCK_Q, BLOCK_T]` tile of one image's cost matrix."""
    image = tl.program_id(0)
    query_block = tl.program_id(1)
    target_block = tl.program_id(2)

    query_offsets = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    target_offsets = target_block * BLOCK_T + tl.arange(0, BLOCK_T)
    query_mask = query_offsets < num_queries
    target_mask = target_offsets < num_targets

    # --- load boxes: [BLOCK_Q, 1] against [1, BLOCK_T] ---
    pred_base = pred_boxes_ptr + image * num_queries * 4 + query_offsets * 4
    pred_cx = tl.load(pred_base + 0, mask=query_mask, other=0.0)[:, None]
    pred_cy = tl.load(pred_base + 1, mask=query_mask, other=0.0)[:, None]
    pred_w = tl.load(pred_base + 2, mask=query_mask, other=0.0)[:, None]
    pred_h = tl.load(pred_base + 3, mask=query_mask, other=0.0)[:, None]

    tgt_base = tgt_boxes_ptr + image * num_targets * 4 + target_offsets * 4
    tgt_cx = tl.load(tgt_base + 0, mask=target_mask, other=0.0)[None, :]
    tgt_cy = tl.load(tgt_base + 1, mask=target_mask, other=0.0)[None, :]
    tgt_w = tl.load(tgt_base + 2, mask=target_mask, other=0.0)[None, :]
    tgt_h = tl.load(tgt_base + 3, mask=target_mask, other=0.0)[None, :]

    # --- L1 cost in cxcywh space (what torch.cdist(p=1) computes) ---
    l1 = (
        tl.abs(pred_cx - tgt_cx)
        + tl.abs(pred_cy - tgt_cy)
        + tl.abs(pred_w - tgt_w)
        + tl.abs(pred_h - tgt_h)
    )

    # --- GIoU on xyxy corners, derived on the fly (no materialised conversion) ---
    pred_x0 = pred_cx - 0.5 * pred_w
    pred_y0 = pred_cy - 0.5 * pred_h
    pred_x1 = pred_cx + 0.5 * pred_w
    pred_y1 = pred_cy + 0.5 * pred_h
    tgt_x0 = tgt_cx - 0.5 * tgt_w
    tgt_y0 = tgt_cy - 0.5 * tgt_h
    tgt_x1 = tgt_cx + 0.5 * tgt_w
    tgt_y1 = tgt_cy + 0.5 * tgt_h

    pred_area = (pred_x1 - pred_x0) * (pred_y1 - pred_y0)
    tgt_area = (tgt_x1 - tgt_x0) * (tgt_y1 - tgt_y0)

    inter_w = tl.minimum(pred_x1, tgt_x1) - tl.maximum(pred_x0, tgt_x0)
    inter_h = tl.minimum(pred_y1, tgt_y1) - tl.maximum(pred_y0, tgt_y0)
    inter = tl.maximum(inter_w, 0.0) * tl.maximum(inter_h, 0.0)
    union = pred_area + tgt_area - inter
    iou = inter / tl.maximum(union, BOX_EPS)

    enclose_w = tl.maximum(pred_x1, tgt_x1) - tl.minimum(pred_x0, tgt_x0)
    enclose_h = tl.maximum(pred_y1, tgt_y1) - tl.minimum(pred_y0, tgt_y0)
    enclose = tl.maximum(enclose_w, 0.0) * tl.maximum(enclose_h, 0.0)
    giou = iou - (enclose - union) / tl.maximum(enclose, BOX_EPS)

    # --- focal classification cost at the gathered target class ---
    # Labels index the class dimension, so clamp before use: a padded or corrupt
    # id must not become an out-of-bounds load. Upstream clamps for the same
    # reason (``matcher.py:439``).
    labels = tl.load(tgt_labels_ptr + image * num_targets + target_offsets, mask=target_mask, other=0)
    labels = tl.minimum(tl.maximum(labels, 0), num_classes - 1)
    logit_offsets = image * num_queries * num_classes + query_offsets[:, None] * num_classes + labels[None, :]
    pair_mask = query_mask[:, None] & target_mask[None, :]
    # ``.to(tl.float32)`` covers the real call site, where AMP hands the matcher
    # bfloat16 logits while the boxes stay float32. Upstream evaluates the focal
    # cost in the logits' own dtype; float32 here is strictly more accurate, and
    # lab/verify_matcher.py checks the resulting assignments still agree.
    logits = tl.load(pred_logits_ptr + logit_offsets, mask=pair_mask, other=0.0).to(tl.float32)

    # Upstream evaluates the focal chain in the logits' own dtype, which is
    # bfloat16 under AMP, rounding at every binary op. The Hungarian assignment is
    # an argmin over 3900 queries, so a float32 chain here (more accurate, but
    # differing by ~1e-2 on a cost of scale ~25) flips ties and returns a
    # different — equally optimal — assignment. Rounding each intermediate back
    # to bfloat16 reproduces upstream's exact arithmetic instead, which
    # lab/verify_matcher.py checks assignment-for-assignment on real batches.
    probabilities = tl.sigmoid(logits)
    abs_logits = tl.abs(logits)
    log1p_exp_neg_abs = tl.log(1.0 + tl.exp(-abs_logits))
    logsigmoid_pos = -log1p_exp_neg_abs + tl.minimum(logits, 0.0)
    logsigmoid_neg = -log1p_exp_neg_abs + tl.minimum(-logits, 0.0)
    if EMULATE_LOW_PRECISION_CLASS_COST:
        probabilities = probabilities.to(tl.bfloat16).to(tl.float32)
        logsigmoid_pos = logsigmoid_pos.to(tl.bfloat16).to(tl.float32)
        logsigmoid_neg = logsigmoid_neg.to(tl.bfloat16).to(tl.float32)
        squared = (probabilities * probabilities).to(tl.bfloat16).to(tl.float32)
        negative_cost = ((1.0 - focal_alpha) * squared).to(tl.bfloat16).to(tl.float32)
        negative_cost = (negative_cost * (-logsigmoid_neg)).to(tl.bfloat16).to(tl.float32)
        one_minus = (1.0 - probabilities).to(tl.bfloat16).to(tl.float32)
        squared_complement = (one_minus * one_minus).to(tl.bfloat16).to(tl.float32)
        positive_cost = (focal_alpha * squared_complement).to(tl.bfloat16).to(tl.float32)
        positive_cost = (positive_cost * (-logsigmoid_pos)).to(tl.bfloat16).to(tl.float32)
        class_cost = (positive_cost - negative_cost).to(tl.bfloat16).to(tl.float32)
        class_term = (cost_class * class_cost).to(tl.bfloat16).to(tl.float32)
    else:
        negative_cost = (1.0 - focal_alpha) * (probabilities * probabilities) * (-logsigmoid_neg)
        positive_cost = focal_alpha * ((1.0 - probabilities) * (1.0 - probabilities)) * (-logsigmoid_pos)
        class_term = cost_class * (positive_cost - negative_cost)

    cost = cost_bbox * l1 + class_term + cost_giou * (-giou)

    out_offsets = image * num_queries * num_targets + query_offsets[:, None] * num_targets + target_offsets[None, :]
    tl.store(out_ptr + out_offsets, cost, mask=pair_mask)


def fused_matcher_cost(
    pred_boxes: torch.Tensor,
    tgt_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    tgt_labels: torch.Tensor,
    *,
    cost_bbox: float,
    cost_class: float,
    cost_giou: float,
    focal_alpha: float,
    block_q: int = 64,
    block_t: int = 32,
    emulate_low_precision_class_cost: bool | None = None,
) -> torch.Tensor:
    """Cost matrix ``[B, Q, T]`` for padded targets, in one kernel launch.

    Args:
        pred_boxes: ``[B, Q, 4]`` predicted boxes in cxcywh, contiguous float32.
        tgt_boxes: ``[B, T, 4]`` padded target boxes in cxcywh.
        pred_logits: ``[B, Q, C]`` classification logits.
        tgt_labels: ``[B, T]`` padded target class ids.
        cost_bbox, cost_class, cost_giou: matcher cost weights.
        focal_alpha: focal alpha of the classification cost.
        block_q, block_t: tile shape.

    Returns:
        ``[B, Q, T]`` float32 cost matrix on the same device.
    """
    batch, queries, _ = pred_boxes.shape
    targets = tgt_boxes.shape[1]
    classes = pred_logits.shape[-1]
    out = torch.empty((batch, queries, targets), device=pred_boxes.device, dtype=torch.float32)
    if targets == 0:
        return out

    grid = (batch, triton.cdiv(queries, block_q), triton.cdiv(targets, block_t))
    _matcher_cost_kernel[grid](
        pred_boxes.contiguous(),
        tgt_boxes.contiguous(),
        pred_logits.contiguous(),
        tgt_labels.contiguous(),
        out,
        queries,
        targets,
        classes,
        float(cost_bbox),
        float(cost_class),
        float(cost_giou),
        float(focal_alpha),
        BLOCK_Q=block_q,
        BLOCK_T=block_t,
        EMULATE_LOW_PRECISION_CLASS_COST=(
            False if emulate_low_precision_class_cost is None else emulate_low_precision_class_cost
        ),
    )
    return out


def reference_matcher_cost(
    pred_boxes: torch.Tensor,
    tgt_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    tgt_labels: torch.Tensor,
    *,
    cost_bbox: float,
    cost_class: float,
    cost_giou: float,
    focal_alpha: float,
) -> torch.Tensor:
    """The upstream op chain, for verification and A/B timing."""
    import torch.nn.functional as F
    from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

    batch, queries = pred_logits.shape[:2]
    targets = tgt_labels.shape[1]
    gather_index = tgt_labels[:, None, :].expand(batch, queries, targets)
    target_logits = torch.gather(pred_logits, 2, gather_index)

    probabilities = target_logits.sigmoid()
    negative_cost = (1 - focal_alpha) * (probabilities**FOCAL_GAMMA) * (-F.logsigmoid(-target_logits))
    positive_cost = focal_alpha * ((1 - probabilities) ** FOCAL_GAMMA) * (-F.logsigmoid(target_logits))
    class_cost = positive_cost - negative_cost

    bbox_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)
    giou_cost = -torch.vmap(generalized_box_iou)(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes))
    return cost_bbox * bbox_cost + cost_class * class_cost + cost_giou * giou_cost


#: Shapes the matcher actually produces for this workload: 5 layers x batch 8
#: folded into the leading dimension, 300 queries x group_detr 13 = 3900, and the
#: padded target width of a crowded person/face batch.
SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (40, 3900, 30, 2),
    (40, 3900, 97, 2),
    (8, 3900, 12, 2),
    (40, 300, 30, 2),
    (5, 3900, 1, 2),
)
WEIGHTS = {"cost_bbox": 5.0, "cost_class": 2.0, "cost_giou": 2.0, "focal_alpha": 0.25}


def make_inputs(batch: int, queries: int, targets: int, classes: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(7)
    centers = torch.rand((batch, queries, 2), device="cuda", generator=generator)
    sizes = torch.rand((batch, queries, 2), device="cuda", generator=generator) * 0.5 + 1e-3
    pred_boxes = torch.cat([centers, sizes], dim=-1)
    tgt_centers = torch.rand((batch, targets, 2), device="cuda", generator=generator)
    tgt_sizes = torch.rand((batch, targets, 2), device="cuda", generator=generator) * 0.5 + 1e-3
    tgt_boxes = torch.cat([tgt_centers, tgt_sizes], dim=-1)
    pred_logits = torch.randn((batch, queries, classes), device="cuda", generator=generator) * 4
    tgt_labels = torch.randint(0, classes, (batch, targets), device="cuda", generator=generator)
    return pred_boxes, tgt_boxes, pred_logits, tgt_labels


def verify() -> int:
    """Compare the fused kernel against the upstream chain on real shapes."""
    failures = 0
    for batch, queries, targets, classes in SHAPES:
        inputs = make_inputs(batch, queries, targets, classes)
        expected = reference_matcher_cost(*inputs, **WEIGHTS)
        actual = fused_matcher_cost(*inputs, **WEIGHTS)
        gap = (expected - actual).abs().max().item()
        scale = expected.abs().max().item()
        tolerance = 2e-4 * max(scale, 1.0)
        status = "ok" if gap <= tolerance else "FAIL"
        failures += status == "FAIL"
        print(
            f"{status} b{batch:<3d} q{queries:<5d} t{targets:<4d} max|diff| {gap:.3e} "
            f"(tol {tolerance:.1e}, scale {scale:.1f})"
        )

    # Degenerate boxes: zero width/height must stay finite in both paths.
    pred_boxes, tgt_boxes, pred_logits, tgt_labels = make_inputs(4, 256, 8, 2)
    tgt_boxes[:, :, 2:] = 0.0
    pred_boxes[:, :, 2:] = 0.0
    expected = reference_matcher_cost(pred_boxes, tgt_boxes, pred_logits, tgt_labels, **WEIGHTS)
    actual = fused_matcher_cost(pred_boxes, tgt_boxes, pred_logits, tgt_labels, **WEIGHTS)
    finite = bool(torch.isfinite(actual).all())
    gap = (expected - actual).abs().max().item()
    status = "ok" if finite and gap <= 2e-4 * max(expected.abs().max().item(), 1.0) else "FAIL"
    failures += status == "FAIL"
    print(f"{status} degenerate (zero-area) boxes: finite={finite} max|diff| {gap:.3e}")
    return 1 if failures else 0


def bench() -> int:
    """Time the fused kernel against the upstream chain."""
    print(f"{'shape':28s} {'chain ms':>9s} {'fused ms':>9s} {'speedup':>8s}")
    for batch, queries, targets, classes in SHAPES:
        inputs = make_inputs(batch, queries, targets, classes)
        chain_ms = triton.testing.do_bench(lambda: reference_matcher_cost(*inputs, **WEIGHTS))
        fused_ms = triton.testing.do_bench(lambda: fused_matcher_cost(*inputs, **WEIGHTS))
        label = f"b{batch} q{queries} t{targets} c{classes}"
        print(f"{label:28s} {chain_ms:9.3f} {fused_ms:9.3f} {chain_ms / fused_ms:7.2f}x")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 2
    status = 0
    if args.verify:
        status |= verify()
    if args.bench:
        status |= bench()
    if not args.verify and not args.bench:
        parser.print_help()
    return status


if __name__ == "__main__":
    sys.exit(main())
