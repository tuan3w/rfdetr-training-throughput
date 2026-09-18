#!/usr/bin/env python3
"""Attribute every host-device synchronisation in a training step to its call site.

Nsight counts ~430 ``cudaStreamSynchronize`` calls per step (~99 ms of host
blocking) but cannot say which Python line asks for them. PyTorch can:
``torch.cuda.set_sync_debug_mode("warn")`` raises a warning on every
synchronising operation, and the warning carries the stack that triggered it.

This runs a few real training steps with that mode on, groups the warnings by the
first frame outside torch/lightning, and prints the ranked call sites. Anything
near the top is either an ``.item()``/``.cpu()``/``.tolist()`` on the hot path or
a device-to-host copy that could stay on the device.

    uv run python -m lab.sync_probe --dataset-dir data/subset --steps 3
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from lab.configs import (
    apply_runtime_settings,
    build_configs,
    build_trainer_without_checkpoints,
    config_parser,
    configure_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser()])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scratch/sync"))
    parser.add_argument("--steps", type=int, default=3, help="Training batches to observe.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Steps run before recording.")
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


ARGS = parse_args()
configure_environment(ARGS)

import torch  # noqa: E402
from pytorch_lightning import Callback  # noqa: E402

#: Frames from these packages are plumbing; the interesting frame is the first
#: one outside them.
_SKIP_MARKERS = ("/torch/", "/pytorch_lightning/", "/lightning_fabric/", "warnings.py", "lab/sync_probe.py")


def attribute(stack: list[traceback.FrameSummary]) -> str:
    """Name the shallowest frame that is not framework plumbing."""
    for frame in reversed(stack):
        if not any(marker in frame.filename for marker in _SKIP_MARKERS):
            location = Path(frame.filename)
            short = "/".join(location.parts[-2:])
            return f"{short}:{frame.lineno} in {frame.name}()  |  {(frame.line or '').strip()[:70]}"
    frame = stack[-1]
    return f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}()"


def main() -> int:
    args = ARGS
    apply_runtime_settings(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config, train_config = build_configs(args)

    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule

    sites: Counter[str] = Counter()
    recording = {"on": False}

    def showwarning(message: Any, category: Any, filename: str, lineno: int, file: Any = None, line: Any = None) -> None:
        if recording["on"] and "synchroniz" in str(message).lower() or (
            recording["on"] and "sync" in str(message).lower()
        ):
            sites[attribute(traceback.extract_stack())] += 1

    warnings.showwarning = showwarning
    warnings.simplefilter("always")

    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)

    class Control(Callback):
        def __init__(self, warmup: int, steps: int) -> None:
            self.warmup = warmup
            self.steps = steps
            self.seen = 0

        def on_train_batch_start(self, trainer: Any, *rest: Any, **kwargs: Any) -> None:
            if self.seen == self.warmup:
                torch.cuda.set_sync_debug_mode("warn")
                recording["on"] = True

        def on_train_batch_end(self, trainer: Any, *rest: Any, **kwargs: Any) -> None:
            self.seen += 1
            if self.seen >= self.warmup + self.steps:
                torch.cuda.set_sync_debug_mode("default")
                recording["on"] = False
                trainer.should_stop = True

    trainer = build_trainer_without_checkpoints(
        train_config,
        model_config,
        limit_train_batches=args.warmup_steps + args.steps + 1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.callbacks.append(Control(args.warmup_steps, args.steps))
    trainer.fit(module, datamodule)
    torch.cuda.set_sync_debug_mode("default")

    total = sum(sites.values())
    print(f"\nsynchronising operations over {args.steps} step(s): {total} ({total / max(1, args.steps):.0f} per step)\n")
    print(f"{'count':>7s} {'per step':>9s}  call site")
    for site, count in sites.most_common(args.top):
        print(f"{count:7d} {count / max(1, args.steps):9.1f}  {site}")
    if not sites:
        print("none recorded — the warning hook saw nothing, check the torch version's message text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
