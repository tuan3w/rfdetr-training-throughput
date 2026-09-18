#!/usr/bin/env python3
"""Attribute the inter-batch gap: real host work, or the host waiting for the GPU.

``lab/gap.py`` measures ~42 ms per microbatch between one batch ending and the
next transfer starting, and shows it is not the dataloader: replaying cached
batches with no workers leaves the gap unchanged. Two very different things look
identical from the outside:

* **Host work** -- Lightning's loop, the optimiser step, the EMA update, logging.
  Fixing it means doing less of it.
* **Blocking waits** -- the host running ahead and then stalling in a sync
  (``.item()``, ``.cpu()``, ``synchronize()``) until the GPU drains. Then the gap
  is really GPU time that has been mis-attributed, and the fix is to make the
  device work smaller, not the host work.

This times every blocking call site and the individual loop phases over a real
fit, so the gap can be split between the two.

    uv run python -m lab.hostgap --dataset-dir data/subset --steps 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from collections import defaultdict
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/scratch/hostgap"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


ARGS = parse_args()
configure_environment(ARGS)

import torch  # noqa: E402
from pytorch_lightning import Callback  # noqa: E402

#: Wall time spent inside each blocking primitive, per measured step.
BLOCKING: dict[str, list[float]] = defaultdict(list)
#: Wall time between named loop phases, per measured step.
PHASES: dict[str, list[float]] = defaultdict(list)
#: Blocking wall time charged to the call site that asked for it.
SITES: dict[str, float] = defaultdict(float)
#: Number of blocking calls per site.
SITE_CALLS: dict[str, int] = defaultdict(int)

_SKIP = ("/torch/", "/pytorch_lightning/", "/lightning_fabric/", "lab/hostgap.py", "/lightning_utilities/")


def short(frame: traceback.FrameSummary) -> str:
    """Render a frame as ``package/module.py:line function()``."""
    location = "/".join(Path(frame.filename).parts[-2:])
    return f"{location}:{frame.lineno} {frame.name}()"


def blame() -> str:
    """Name the direct caller, plus the nearest non-framework frame above it.

    The direct caller is what matters for a blocking scalar read -- it is often
    inside Lightning or torch itself rather than in model code -- but on its own
    it does not say which part of the step triggered it, so the nearest frame
    outside the frameworks is reported as context.
    """
    stack = [frame for frame in traceback.extract_stack() if "lab/hostgap.py" not in frame.filename]
    if not stack:
        return "unknown"
    caller = short(stack[-1])
    for frame in reversed(stack):
        if not any(marker in frame.filename for marker in _SKIP):
            context = short(frame)
            if context != caller:
                return f"{caller}  <- {context}"
            break
    return caller
RECORDING = {"on": False}
_current: dict[str, float] = defaultdict(float)


def instrument_blocking() -> None:
    """Time the operations that can block the host on the device."""
    original_sync = torch.cuda.synchronize
    original_item = torch.Tensor.item
    original_cpu = torch.Tensor.cpu
    original_numpy = torch.Tensor.numpy

    def timed(name: str, function: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not RECORDING["on"]:
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - started) * 1e3
                _current[name] += elapsed
                # Walking the stack is expensive, so only blame calls that
                # actually blocked; sub-100us calls are not what we are hunting.
                if elapsed > 0.1:
                    site = blame()
                    SITES[site] += elapsed
                    SITE_CALLS[site] += 1

        return wrapper

    torch.cuda.synchronize = timed("synchronize", original_sync)
    torch.Tensor.item = timed("item", original_item)  # type: ignore[method-assign]
    torch.Tensor.cpu = timed("cpu", original_cpu)  # type: ignore[method-assign]
    torch.Tensor.numpy = timed("numpy", original_numpy)  # type: ignore[method-assign]


def main() -> int:
    args = ARGS
    apply_runtime_settings(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config, train_config = build_configs(args)

    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule

    instrument_blocking()
    marks: dict[str, float] = {}
    gaps: list[float] = []
    computes: list[float] = []
    seen = {"steps": 0}

    def mark(name: str) -> None:
        marks[name] = time.perf_counter()

    def span(start: str, end: str, into: str) -> None:
        if start in marks and end in marks and RECORDING["on"]:
            _current[into] += (marks[end] - marks[start]) * 1e3

    original_transfer = RFDETRDataModule.transfer_batch_to_device

    def timed_transfer(self: Any, batch: Any, device: Any, dataloader_idx: int) -> Any:
        mark("transfer_start")
        if "batch_end" in marks and RECORDING["on"]:
            gaps.append((marks["transfer_start"] - marks["batch_end"]) * 1e3)
        return original_transfer(self, batch, device, dataloader_idx)

    RFDETRDataModule.transfer_batch_to_device = timed_transfer

    class Probe(Callback):
        def on_train_batch_start(self, trainer: Any, *rest: Any, **kwargs: Any) -> None:
            if seen["steps"] == args.warmup_steps:
                RECORDING["on"] = True
            mark("batch_start")
            span("transfer_start", "batch_start", "after_transfer_to_batch_start")

        def on_before_backward(self, trainer: Any, pl_module: Any, loss: Any) -> None:
            mark("before_backward")

        def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
            mark("before_optimizer")

        def on_before_zero_grad(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
            mark("before_zero_grad")
            # on_before_optimizer_step only fires on accumulation boundaries, so a
            # stale mark from an earlier microbatch would make this span nonsense.
            if marks.get("before_optimizer", 0.0) > marks.get("batch_start", 0.0):
                span("before_optimizer", "before_zero_grad", "optimizer_step")
                marks.pop("before_optimizer")

        def on_train_batch_end(self, trainer: Any, *rest: Any, **kwargs: Any) -> None:
            mark("batch_end_pre")
            if RECORDING["on"]:
                computes.append((marks["batch_end_pre"] - marks["batch_start"]) * 1e3)
                for name, value in _current.items():
                    BLOCKING[name].append(value) if name in {
                        "synchronize",
                        "item",
                        "cpu",
                        "numpy",
                    } else PHASES[name].append(value)
                _current.clear()
            mark("batch_end")
            seen["steps"] += 1
            if len(computes) >= args.steps:
                trainer.should_stop = True

    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer_without_checkpoints(
        train_config,
        model_config,
        limit_train_batches=args.warmup_steps + args.steps + 2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.callbacks.append(Probe())
    trainer.fit(module, datamodule)
    RFDETRDataModule.transfer_batch_to_device = original_transfer

    if not computes:
        print("no measured steps", file=sys.stderr)
        return 1

    compute = statistics.median(computes)
    gap = statistics.median(gaps) if gaps else 0.0
    print(f"\nsteps measured        : {len(computes)}")
    print(f"batch_start -> end    : {compute:7.2f} ms")
    print(f"end -> next transfer  : {gap:7.2f} ms   <- the gap under investigation")
    print("\nblocking calls (median ms per microbatch, anywhere in the step):")
    for name in ("synchronize", "item", "cpu", "numpy"):
        if BLOCKING[name]:
            print(f"  {name:12s} {statistics.median(BLOCKING[name]):7.2f}")
    print(f"\nblocking time by call site (total over {len(computes)} microbatches):")
    print(f"{'ms total':>9s} {'ms/step':>8s} {'calls':>6s}  site")
    for site, total in sorted(SITES.items(), key=lambda item: -item[1])[:8]:
        print(f"{total:9.1f} {total / len(computes):8.2f} {SITE_CALLS[site]:6d}  {site}")

    print("\nloop phases (median ms per microbatch):")
    for name, values in PHASES.items():
        if values:
            print(f"  {name:32s} {statistics.median(values):7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
