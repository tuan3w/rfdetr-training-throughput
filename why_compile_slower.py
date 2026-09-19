#!/usr/bin/env python3
"""Explain why a compiled DINOv2 encoder is slower than eager on this GPU.

``lab/bench_backbone.py`` measures the encoder at 36.5 ms eager and 60.6 ms
compiled (0.60x), reproducibly. A first look said bf16 cutlass GEMM time roughly
doubles, which on its own makes no sense: Inductor does not rewrite the GEMMs,
it calls the same libraries on the same shapes.

So this compares the two runs kernel by kernel -- per-kernel time *and call
count*, plus the aten-level GEMM shapes -- to separate the possibilities:

* more GEMM calls (a decomposition adding matmuls, or recompute in backward),
* the same calls getting slower (layout or algorithm choice),
* time moving into Triton kernels that replace something cheaper.

    uv run python -m lab.why_compile_slower
"""

from __future__ import annotations

import argparse
import collections
import sys
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def classify(name: str) -> str:
    """Group a kernel name into a comparable bucket."""
    lowered = name.lower()
    if "cutlass" in lowered or "gemm" in lowered or "gemv" in lowered:
        return "GEMM (cutlass/cublas)"
    if "flash" in lowered:
        return "attention (flash)"
    if "efficient_attention" in lowered or "mem_eff" in lowered:
        return "attention (mem-efficient)"
    if name.startswith("triton_"):
        return "triton (inductor)"
    if "elementwise" in lowered or "vectorized" in lowered:
        return "elementwise (aten)"
    if "layer_norm" in lowered or "gammabeta" in lowered:
        return "layer norm (aten)"
    if "reduce" in lowered or "sum" in lowered:
        return "reduction (aten)"
    if "memcpy" in lowered or "memset" in lowered:
        return "copy/memset"
    return "other"


def main() -> int:
    args = parse_args()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.set_float32_matmul_precision("high")

    from rfdetr.detr import RFDETRMedium

    detector = RFDETRMedium(
        device="cuda", num_classes=2, resolution=args.resolution, gradient_checkpointing=False
    )
    encoder = detector.model.model.backbone[0].encoder.cuda().train()
    pixels = torch.randn((args.batch_size, 3, args.resolution, args.resolution), device="cuda")

    def step(module: Any) -> None:
        module.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = module(pixels)
        output = getattr(features, "last_hidden_state", None)
        if output is None:
            output = features[0] if isinstance(features, (list, tuple)) else features
        getattr(output, "tensors", output).float().pow(2).mean().backward()

    def measure(module: Any, label: str) -> tuple[dict[str, tuple[float, int]], dict[str, tuple[float, int]]]:
        for _ in range(3):
            step(module)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            step(module)
            torch.cuda.synchronize()

        buckets: dict[str, list[float | int]] = collections.defaultdict(lambda: [0.0, 0])
        matmuls: dict[str, list[float | int]] = collections.defaultdict(lambda: [0.0, 0])
        for event in prof.key_averages(group_by_input_shape=True):
            device_time = event.self_device_time_total / 1e3
            if device_time <= 0:
                continue
            if event.key.startswith("aten::"):
                if any(op in event.key for op in ("mm", "bmm", "matmul", "linear")):
                    shape = str(event.input_shapes)[:58]
                    entry = matmuls[f"{event.key} {shape}"]
                    entry[0] += device_time
                    entry[1] += event.count
                continue
            entry = buckets[classify(event.key)]
            entry[0] += device_time
            entry[1] += event.count
        print(f"\n=== {label}: total device time {sum(v[0] for v in buckets.values()):.1f} ms")
        print(f"{'bucket':26s} {'ms':>8s} {'calls':>7s}")
        for name, (milliseconds, calls) in sorted(buckets.items(), key=lambda item: -item[1][0]):
            print(f"{name:26s} {milliseconds:8.2f} {calls:7d}")
        return (
            {k: (v[0], v[1]) for k, v in buckets.items()},
            {k: (v[0], v[1]) for k, v in matmuls.items()},
        )

    eager_buckets, eager_mm = measure(encoder, "eager")
    compiled_buckets, compiled_mm = measure(torch.compile(encoder, dynamic=False), "compiled")

    print("\n=== bucket deltas (compiled - eager)")
    print(f"{'bucket':26s} {'d ms':>9s} {'d calls':>9s}")
    for name in sorted(set(eager_buckets) | set(compiled_buckets)):
        was_ms, was_calls = eager_buckets.get(name, (0.0, 0))
        now_ms, now_calls = compiled_buckets.get(name, (0.0, 0))
        print(f"{name:26s} {now_ms - was_ms:+9.2f} {now_calls - was_calls:+9d}")

    print("\n=== aten matmul rows, by device time (eager)")
    for name, (milliseconds, calls) in sorted(eager_mm.items(), key=lambda item: -item[1][0])[: args.top]:
        print(f"{milliseconds:8.2f} ms {calls:5d}x  {name}")
    print("\n=== aten matmul rows, by device time (compiled)")
    for name, (milliseconds, calls) in sorted(compiled_mm.items(), key=lambda item: -item[1][0])[: args.top]:
        print(f"{milliseconds:8.2f} ms {calls:5d}x  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
