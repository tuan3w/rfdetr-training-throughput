#!/usr/bin/env python3
"""Paired A/B benchmark of two source trees of the same package.

A single before/after pair is not evidence when the run-to-run spread is larger
than the effect: the first attempt at measuring the assignment change read 24.31
and 26.47 img/s for the *same* code. This alternates the two trees so thermal
drift and background load fall on both sides equally, then reports each side's
median and the paired differences.

The two variants are selected with ``PYTHONPATH`` over separate git worktrees,
deliberately not by stashing in one tree: a run interrupted mid-round leaves the
change in the stash and the worktree looking clean, which is a good way to lose
work.

    git worktree add /tmp/rf-develop <base-commit>
    uv run python -m lab.bench_pr --before-src /tmp/rf-develop/src \
        --after-src ~/workspace/rf-detr/src --python ~/workspace/rf-detr/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

#: Parses the throughput out of a cell's stdout.
METRIC = re.compile(r"images_per_s=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-src", type=Path, help="Baseline tree's `src` directory.")
    parser.add_argument("--after-src", type=Path, help="Changed tree's `src` directory.")
    parser.add_argument(
        "--before-patch",
        help="Patch list for the baseline side; use this instead of the tree arguments to A/B lab patches.",
    )
    parser.add_argument("--after-patch", help="Patch list for the changed side.")
    parser.add_argument("--python", type=Path, required=True, help="Interpreter with the package's dependencies.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/subset"))
    parser.add_argument("--rounds", type=int, default=3, help="Before/after pairs to run.")
    parser.add_argument("--compile", type=int, default=0, help="Compile the model; the sync removal only pays where the host is not already ahead.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument(
        "--multi-scale",
        type=int,
        default=0,
        help="Left off by default: varying input shapes dominate the variance and hide a small effect.",
    )
    return parser.parse_args()


def run_cell(args: argparse.Namespace, label: str, source: Path | None, patch: str) -> float:
    """Measure one cell and return images per second.

    ``source`` selects a package tree through ``PYTHONPATH`` when comparing two
    checkouts; ``patch`` selects the lab patch list when comparing configurations
    of one checkout.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as results:
        command = [
            str(args.python),
            "-m",
            "lab.cell",
            "--name",
            label,
            "--mode",
            "step",
            "--dataset-dir",
            str(args.dataset_dir),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--compile",
            str(args.compile),
            "--resolution",
            str(args.resolution),
            "--multi-scale",
            str(args.multi_scale),
            "--patch",
            patch,
            "--result-file",
            results.name,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "." if source is None else f".:{source}", "PATH": "/usr/bin:/bin"},
        )
    found = METRIC.findall(completed.stdout)
    if not found:
        raise RuntimeError(f"no metric in output of {label}: {completed.stdout[-400:]}\n{completed.stderr[-400:]}")
    return float(found[-1])


def main() -> int:
    args = parse_args()
    comparing_trees = args.before_src is not None and args.after_src is not None
    if comparing_trees:
        for source in (args.before_src, args.after_src):
            if not (source / "rfdetr").is_dir():
                print(f"{source} does not look like a package source tree", file=sys.stderr)
                return 1
    elif args.before_patch is None or args.after_patch is None:
        print("pass either --before-src/--after-src or --before-patch/--after-patch", file=sys.stderr)
        return 1

    before_patch = args.before_patch or ""
    after_patch = args.after_patch or ""
    before: list[float] = []
    after: list[float] = []
    for round_index in range(args.rounds):
        baseline = run_cell(args, f"pr-before-{round_index}", args.before_src, before_patch)
        changed = run_cell(args, f"pr-after-{round_index}", args.after_src, after_patch)
        before.append(baseline)
        after.append(changed)
        delta = (changed - baseline) / baseline * 100
        print(f"round {round_index}: before {baseline:6.2f}  after {changed:6.2f}  {delta:+5.2f}%", flush=True)

    deltas = [(changed - baseline) / baseline * 100 for baseline, changed in zip(before, after)]
    print(f"\nbefore median {statistics.median(before):6.2f} img/s  (spread {max(before) - min(before):.2f})")
    print(f"after  median {statistics.median(after):6.2f} img/s  (spread {max(after) - min(after):.2f})")
    print(f"paired deltas {', '.join(f'{d:+.2f}%' for d in deltas)}  median {statistics.median(deltas):+.2f}%")
    print(f"\n{json.dumps({'before': before, 'after': after, 'deltas': deltas})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
