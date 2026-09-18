# Making an RF-DETR fine-tune 2.6× faster

**[→ Interactive report with charts](https://tuan3w.github.io/rfdetr-training-throughput/)**

A single-GPU [RF-DETR](https://github.com/roboflow/rf-detr) Medium person/face fine-tune ran at
**10.4 images/second** — 4h23m per epoch, ~18 days for its configured 100 epochs. After a measurement-driven
optimisation loop it runs at **33.6 img/s (2.63×)**, without changing the training recipe: same resolution,
same multi-scale set, same effective batch, same augmentation stack, same EMA.

| | before | after |
|---|---|---|
| throughput | 12.81 img/s | **33.63 img/s** |
| step time | 624 ms | **238 ms** |
| epoch (164k images) | 4h23m | **~1h21m** |
| peak memory | 5.91 GiB | 6.49 GiB |

Everything below was measured, never assumed. 24 experiments were logged with a keep/discard decision; 7
survived. The negative results are included because they are the expensive part.

---

## The starting point

The script carried a stack of workarounds added after repeated `Xid 31` MMU faults on a shared A100:
math-only SDPA, a SciPy-forced Hungarian matcher, non-fused AdamW, gradient checkpointing, and the classic
CUDA allocator.

**During this work the job crashed again — `Xid 31` — while running the full workaround stack.** So the
workarounds cost 1.6–2× throughput and did not prevent the fault. The Xid history spanned both tenants of the
shared GPU, pointing at the torch 2.14+cu130 / driver 595 stack or the co-tenancy rather than at any of the
three flags.

## Method

A harness that measures the **real** PyTorch Lightning training loop — dataloader → augmentation → AMP
forward → matcher/criterion → backward → optimizer → EMA — on a fixed 3,000-image subset, reporting the
median of 2–3 independent processes of 40 measured steps.

Two properties made the results trustworthy:

1. **Fidelity.** Running the *live* configuration through the harness on the production A100 measured
   **10.04 img/s** against the real job's observed 10.4 img/s.
2. **An honesty guard.** `lab/guard.py` fails any run that reaches its number by weakening training —
   resolution, multi-scale set, effective batch (≥32), augmentation preset and EMA are pinned. Throughput is
   trivially gamed by making the model worse; the guard makes that a failed run rather than a win.

Development and measurement happened on a local RTX 5060 Ti (exclusive, reproducible). The production A100
was used only for cross-checks, because a co-tenant held 22.5 GiB and pinned it at 100% utilisation.

## What worked

| change | gain | why |
|---|---|---|
| `sdpa=auto` | **+61.5%** | The math-only SDPA workaround was the single biggest cost: 24× slower than fused attention on the windowed backbone shapes, 7.8× on the global ones. Also freed 1.4 GiB. |
| `grad_ckpt=0` | +6.1% | Recomputing the backbone stops being free once attention is fused. |
| `torch.compile` | +13.6% | Inductor fusion; −0.8 GiB. One-off ~11 min warm-up (~3 min with a warm cache). |
| fused AdamW | +1.3% | Noise locally; 1.20× on the launch-bound A100. |
| **fused Triton matcher kernel** | **+20.0%** | See below. |
| **parallel assignment solve** | **+13.3%** | See below. |

### Kernel 1 — the Hungarian cost matrix (+20%)

The PyTorch profiler found one operator owning **15.3% of all device time in 5 calls per step**:
`aten::_cdist_forward`. rf-detr builds the matcher's cost matrix as a chain of separate kernels — class
gather, focal cost, `cdist` L1, two box conversions, a vmapped GIoU, then three multiplies and two adds —
and every pass reads and writes a full `[batch, 3900 queries, targets]` tensor. `torch.cdist(p=1)` on a
4-wide box dimension also lands on a generic kernel.

One Triton kernel now computes the whole thing in a single launch: **174× faster standalone** on the
production shape (25.4 ms → 0.145 ms). rf-detr ships no custom kernels at all, so this was unoptimised
ground.

### Kernel 2 — the surprise: the solver belongs on the CPU (+13.3%)

NVTX ranges in an Nsight trace showed the host spending **102 ms per microbatch inside the criterion** — as
much as the entire backward pass — because the SciPy assignment solve drains the GPU pipeline five times per
step.

The instinctive fix was to move the solve to the GPU. rf-detr supports that (a Triton-backed batched solver),
and it **lost 35%**: it does remove the sync (device idle 19.0% → 12.5%) but adds ~10 ms per call × 5 calls.
The workload is ~520 *tiny* assignment problems per step (5 decoder layers × 8 images × 13 query groups,
`[300 × T]` with T between 1 and 60) — latency-bound, with almost no parallel width per problem.

Solving those same problems on an **8-thread pool** instead gained 13.3%, and returns bit-identical
matchings. A thread sweep put the optimum at 8 (4 → 29.1, 8 → 30.1, 16 → 28.9, 24 → 29.4 img/s): beyond 8 it
contends with the 16 dataloader workers.

## How the bottleneck was located

Each tool was escalated to only after the previous one was exhausted:

| tool | measurement | conclusion |
|---|---|---|
| `torch.profiler` | `aten::_cdist_forward` 37.6 ms/step, 15.3% of device time | → kernel 1 |
| `torch.profiler` | `gpu_busy` 0.49, `cudaStreamSynchronize` 41% of host time | host-bound, not kernel-bound |
| Nsight Systems | device busy 69.9%; **25.7% of all idle in 104k gaps under 50 µs** | 3.8 µs launch latency × 4,835 launches/step |
| Nsight NVTX | criterion 102 ms vs backward 106 ms per microbatch; EMA 5.9 µs | → kernel 2 |
| gap decomposition | 218 ms compute + 47 ms gap, of which H2D transfer is **2.9 ms** | the rest is framework overhead, not data |
| loader-only mode | ceiling 239 img/s vs ~34 consumed | 8× headroom: augmentation tuning cannot pay |

That last row killed an entire branch of plausible-sounding work. The augmentation stack, worker count and
prefetch depth simply cannot matter when the input pipeline delivers 8× what the loop consumes.

## What did not work

| attempt | result |
|---|---|
| GPU assignment solver | **−35%** — solve kernel costs more than the sync it removes |
| Fused Triton deformable attention (fwd + all 3 grads) | +2.9% eager, **−4.0% under `compile`** — Inductor already fuses that chain; a custom autograd function is opaque to it |
| CUDA graphs (rf-detr `develop`) | **OOM** — 12.3 GiB of private graph pools after 2 of 8 multi-scale shapes |
| Batch 12 / 16 (tested 3×) | −1% to −8.6% locally; **+12.7% on the A100** (108 SMs vs 24) |
| Pinned-memory host transfer (2 variants) | −16% to +2%, all noise |
| Compiling the criterion | no effect; compiling all of it hits an internal Dynamo bug |
| Fusing the 6 augmentation wrappers into one | −2% — hidden by loader headroom |
| `pack_targets=0` | −5.4% — the existing packing is a real win |
| More workers / prefetch, thread counts, EMA interval, expandable segments | all within noise |

## Correctness

A faster kernel that is subtly wrong would quietly damage training, so each was held to what a consumer
observes — the assignment decisions and the loss trajectory — not just to matching floats.

- Triton matcher vs upstream: ≤5e-06 across five production shapes (synthetic); **3.8e-06 on real batches
  locally and exactly 0.0 on the A100**; degenerate zero-area boxes stay finite; non-L1 metrics fall back
  bit-exactly.
- **Assignment decisions: 0 mismatches out of 120 on the A100.** Locally some differ, but only among
  equally optimal solutions — objective gap 2.1e-04 relative.
- **Loss trajectory**, deterministic loader: the kernel deviates 2.9e-02, versus **3.1e-02 between two
  identical unpatched runs**. The kernel's effect is smaller than the training loop's own nondeterminism.
- Parallel solve: matchings bit-identical to the serial loop across 4 shapes, including 520×300×37.
- The benchmark entrypoint refuses to report a number unless all verifiers pass whenever a patch is active.

## Recommended configuration

```
--sdpa-backend auto --fused-optimizer --no-gradient-checkpointing \
--batch-size 16 --grad-accum-steps 2
```

plus `compile=True` on the model constructor and the two accepted patches. Reduce `epochs` from 100 to
25–30 (rf-detr's docs recommend 20–30 above 10k images; this dataset has 164k), and set `eval_interval 5`,
`eval_batch_size 16`, `eval_max_dets 100`. Keep what was already right: the SciPy solver, `pack_targets`,
16 workers, thread pinning, `expandable_segments:False`.

On the shared A100, the stack measured 10.04 → 16.70 img/s (**+66%**) even while time-slicing with the
co-tenant, before the parallel solve existed.

## Where the remaining time goes

Device time is now dominated by real work — cuBLAS GEMMs 68 ms/step (40%), flash-attention backward 12.7 ms.
What is left is **~44 ms/step of Lightning per-microbatch overhead** and **~18 ms/step of kernel-launch
latency** across 4,835 launches. Both would need CUDA graphs, which this workload rules out: one graph memory
pool per multi-scale shape.

## Reproducing

The harness is a small uv project: one file per concern — `cell.py` (measure one configuration),
`guard.py` (honesty invariants), `profile.py` / `nsys_gaps.py` / `nvtx.py` / `gap.py` (attribution),
`kernels.py` and `kernels_msda.py` (Triton kernels with `--verify`/`--bench`), `patches.py` (opt-in
monkeypatches, so the upstream checkout is never modified), `verify_matcher.py` (real-batch equivalence),
`report.py` / `export_report.py` (dashboards).

```bash
uv sync
python lab/make_subset.py <dataset> data/subset --train 3000
uv run python -m lab.kernels --verify --bench      # kernel correctness + speed
uv run python -m lab.cell --mode step --dataset-dir data/subset --steps 40
bash autoresearch.sh                                # full gated benchmark
```

---

*Measured on an RTX 5060 Ti (development) and an A100-SXM4-40GB (cross-check), rf-detr 1.10.1,
torch 2.14.0+cu130, Triton 3.8.*
