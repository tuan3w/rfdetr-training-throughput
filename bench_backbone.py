#!/usr/bin/env python3
"""Measure what a compiled DINOv2 backbone would actually be worth.

``compile=True`` never compiles the model (the positional-embedding resize makes
Inductor reject the graph, and ``suppress_errors=True`` hides it), so the
backbone runs eager. Nsight shows the symptom: ~11,000 tiny elementwise launches
per 1.2 s window at 3-5 us each.

Fixing that is expensive -- the whole-graph compile ran 22 minutes on one frame
without finishing, and there are 8 multi-scale resolutions -- so the question to
answer first is whether a compiled backbone is worth any of it. This times the
backbone alone, forward and backward, eager versus compiled at a single fixed
resolution, which bounds the compile to one graph.

    uv run python -m lab.bench_backbone --resolution 576
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any, Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument(
        "--dynamic",
        type=int,
        default=0,
        help=(
            "Compile with dynamic=True (1) or dynamic=False (0). rfdetr uses dynamic=True so one "
            "graph serves all multi-scale sizes, which is a different codegen regime."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def timed(function: Callable[[], Any], reps: int) -> float:
    """Median milliseconds per call, after a warm-up."""
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)

    from rfdetr.detr import RFDETRMedium

    # The kept configuration, not the stock defaults: with the stock math-only
    # SDPA and gradient checkpointing the backbone measures 1551 ms, which says
    # nothing about the compile question being asked here.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.set_float32_matmul_precision("high")

    detector = RFDETRMedium(
        device="cuda", num_classes=2, resolution=args.resolution, gradient_checkpointing=False
    )
    # The full backbone's forward takes a NestedTensor, a plain dataclass that
    # Dynamo refuses to trace, so compiling it is a no-op (2.4 s "compile", 1.01x).
    # The DINOv2 encoder underneath takes a plain tensor and is the part that
    # holds the transformer blocks, so that is what is measured.
    backbone = detector.model.model.backbone[0].encoder.cuda().train()

    # No gradient w.r.t. the input: training does not need one, and asking for it
    # makes cuDNN compute the patch-embedding conv's input gradient with a
    # grouped-direct algorithm that took 1.466 s in a single call -- 97.6% of an
    # earlier, meaningless 1.5 s measurement.
    pixels = torch.randn((args.batch_size, 3, args.resolution, args.resolution), device="cuda")
    # The backbone consumes a NestedTensor (tensors + padding mask), as built by
    # the datamodule's collate; training never pads, so the mask is all False.


    def step(module: Any) -> Callable[[], Any]:
        def run() -> Any:
            module.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = module(pixels)
            # Every returned feature map, not just the first. This encoder hands
            # back one map per `out_feature_indexes` layer, so a loss on
            # `features[0]` only back-propagates through the layers feeding the
            # earliest map: eager then ran 3 attention backwards against 12
            # forwards, and compared against a compiled graph that computed all
            # 12 the comparison was measuring unequal work.
            maps = features if isinstance(features, (list, tuple)) else [features]
            tensors = [getattr(item, "tensors", item) for item in maps]
            tensors = [item for item in tensors if torch.is_tensor(item)]
            if not tensors:
                raise RuntimeError(f"no tensors in backbone output of type {type(features).__name__}")
            loss = sum(tensor.float().pow(2).mean() for tensor in tensors)
            loss.backward()
            return loss

        return run

    eager_ms = timed(step(backbone), args.reps)
    print(f"eager encoder       : {eager_ms:8.2f} ms  (batch {args.batch_size}, {args.resolution}px)")

    compile_started = time.perf_counter()
    compiled = torch.compile(backbone, dynamic=bool(args.dynamic))
    try:
        step(compiled)()
    except Exception as error:  # noqa: BLE001 - reporting is the point
        print(f"compiled encoder    : FAILED to compile: {type(error).__name__}: {str(error)[:160]}")
        return 1
    compile_seconds = time.perf_counter() - compile_started
    compiled_ms = timed(step(compiled), args.reps)

    print(
        f"compiled encoder    : {compiled_ms:8.2f} ms  ({eager_ms / compiled_ms:.2f}x)  "
        f"dynamic={bool(args.dynamic)}"
    )
    print(f"one-off compile     : {compile_seconds:8.1f} s for this single resolution")
    saving = (eager_ms - compiled_ms) / 1e3
    if saving > 0:
        print(f"break-even          : {compile_seconds / saving:8.0f} steps at one resolution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
