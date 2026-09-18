#!/usr/bin/env python3
"""Reject configurations that buy throughput by weakening training.

The benchmark metric is images/second, which is trivially gamed: drop the
augmentation stack, drop EMA, shrink the resolution, shrink the effective batch
and the number goes up while the resulting model gets worse. This guard fails the
run instead, so every kept improvement is a real efficiency win on a training
recipe that stays equivalent to the live one.

Invariants (all derived from the live A100 config in ``lab/baseline.py``):

* ``resolution`` stays 576 — accuracy-bearing input size.
* ``multi_scale`` stays on, ``expanded_scales`` stays off — same scale set.
* Effective batch (``batch_size * grad_accum``) stays >= 32 — same optimizer window.
* Augmentation keeps geometry *and* photometric ops: preset ``live``, ``live_fused``
  (the same six ops inside one ``Sequential`` container — identical sampling, one
  wrapper instead of six) or ``kornia`` (which cannot express ``HueSaturationValue``).
* ``use_ema`` stays on and ``ema_update_interval`` stays <= 4 — the shipped
  checkpoint is the EMA one.
* ``model`` stays ``medium``.

Free to change: SDPA backend, fused optimizer, gradient checkpointing, matcher
backend, worker/thread/prefetch layout, batch geometry at constant effective
batch, allocator mode, torch.compile, augmentation *backend* and wrapper layout.
"""

from __future__ import annotations

import sys

from lab.baseline import BASELINE

ALLOWED_AUG_PRESETS = {"live", "live_fused", "kornia"}
MIN_EFFECTIVE_BATCH = 32
MAX_EMA_INTERVAL = 4


def violations(config: dict[str, object]) -> list[str]:
    """Return every invariant the config breaks."""
    found: list[str] = []
    if config["model"] != "medium":
        found.append(f"model must stay 'medium', got {config['model']!r}")
    if config["resolution"] != 576:
        found.append(f"resolution must stay 576, got {config['resolution']}")
    if not config["multi_scale"]:
        found.append("multi_scale must stay enabled")
    if config["expanded_scales"]:
        found.append("expanded_scales must stay disabled (it widens the scale set)")

    effective = int(config["batch_size"]) * int(config["grad_accum"])  # type: ignore[arg-type]
    if effective < MIN_EFFECTIVE_BATCH:
        found.append(f"effective batch {effective} < {MIN_EFFECTIVE_BATCH}")

    if config["aug"] not in ALLOWED_AUG_PRESETS:
        found.append(f"aug preset {config['aug']!r} not in {sorted(ALLOWED_AUG_PRESETS)}")
    if config["aug"] == "kornia" and config["aug_backend"] not in {"kornia", "auto"}:
        found.append("aug preset 'kornia' requires aug_backend 'kornia' or 'auto'")
    if config["aug"] == "live" and config["aug_backend"] == "kornia":
        found.append("aug_backend 'kornia' cannot run the 'live' preset (HueSaturationValue unsupported)")

    if not config["use_ema"]:
        found.append("use_ema must stay enabled")
    if int(config["ema_update_interval"]) > MAX_EMA_INTERVAL:  # type: ignore[arg-type]
        found.append(f"ema_update_interval {config['ema_update_interval']} > {MAX_EMA_INTERVAL}")
    return found


def main() -> int:
    found = violations(BASELINE)
    if found:
        print("GUARD FAILED — configuration weakens training:", file=sys.stderr)
        for item in found:
            print(f"  - {item}", file=sys.stderr)
        return 1
    effective = int(BASELINE["batch_size"]) * int(BASELINE["grad_accum"])
    print(
        f"guard ok: medium@{BASELINE['resolution']}, effective batch {effective}, "
        f"aug={BASELINE['aug']}/{BASELINE['aug_backend']}, ema every {BASELINE['ema_update_interval']} step(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
