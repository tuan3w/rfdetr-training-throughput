#!/usr/bin/env python3
"""Count host synchronisations per training step, attributed to their call site.

Throughput on this machine cannot resolve the effect of removing a host round
trip: paired runs of identical code spread by up to 9%. The synchronisation count
can -- it is deterministic, and it is what the change actually alters.

Runs a few real training steps with ``torch.cuda.set_sync_debug_mode("warn")``
and groups the warnings by the innermost frame outside the warnings machinery, so
the numbers can be compared between two source trees (select them with
``PYTHONPATH``).

    PYTHONPATH=.:/tmp/rf-develop/src uv run python -m lab.count_syncs --dataset-dir data/subset
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/scratch/syncs"))
    parser.add_argument("--steps", type=int, default=3, help="Steps counted after warm-up.")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


ARGS = parse_args()
configure_environment(ARGS)

import torch  # noqa: E402
from pytorch_lightning import Callback  # noqa: E402


def innermost_caller() -> str:
    """Name the innermost stack frame that is not the warnings machinery or this module."""
    for frame in reversed(traceback.extract_stack()):
        name = Path(frame.filename).name
        # `warnings.py` up to Python 3.13, `_py_warnings.py` from 3.14.
        if "warnings" in name or name == "count_syncs.py":
            continue
        return f"{'/'.join(Path(frame.filename).parts[-2:])}:{frame.lineno} {frame.name}()"
    return "unknown"


def main() -> int:
    args = ARGS
    apply_runtime_settings(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config, train_config = build_configs(args)

    import rfdetr

    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule

    sites: Counter[str] = Counter()
    recording = {"on": False}

    def record(message: Any, *_args: Any, **_kwargs: Any) -> None:
        """Attribute one synchronisation warning to its caller."""
        if recording["on"] and "synchron" in str(message).lower():
            sites[innermost_caller()] += 1

    warnings.showwarning = record
    warnings.simplefilter("always")

    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)

    class Control(Callback):
        """Turns counting on after warm-up and stops the fit once enough steps are counted."""

        def __init__(self) -> None:
            self.seen = 0

        def on_train_batch_start(self, trainer: Any, *rest: Any, **kwargs: Any) -> None:
            if self.seen == args.warmup_steps:
                torch.cuda.set_sync_debug_mode("warn")
                recording["on"] = True

        def on_train_batch_end(self, trainer: Any, *rest: Any, **kwargs: Any) -> None:
            self.seen += 1
            if self.seen >= args.warmup_steps + args.steps:
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
    trainer.callbacks.append(Control())
    trainer.fit(module, datamodule)
    torch.cuda.set_sync_debug_mode("default")

    total = sum(sites.values())
    print(f"\nrfdetr source: {Path(rfdetr.__file__).parent}")
    print(f"synchronisations over {args.steps} step(s): {total} ({total / max(1, args.steps):.1f} per step)\n")
    print(f"{'per step':>9s}  call site")
    for site, count in sites.most_common(args.top):
        print(f"{count / max(1, args.steps):9.1f}  {site}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
