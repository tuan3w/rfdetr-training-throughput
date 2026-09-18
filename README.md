# Making an RF-DETR fine-tune 3.3× faster

**[→ Interactive report with charts](https://tuan3w.github.io/rfdetr-training-throughput/)** ·
**[→ Full session history](https://tuan3w.github.io/rfdetr-training-throughput/history.html)**

A single-GPU [RF-DETR](https://github.com/roboflow/rf-detr) Medium person/face fine-tune ran at
**10.4 images/second** — 4h23m per epoch, ~18 days for its configured 100 epochs. After a measurement-driven
optimisation loop it runs at **41.9 img/s (3.27×)**, without changing the training recipe: same resolution,
same multi-scale set, same effective batch, same augmentation stack, same EMA.

| | before | after |
|---|---|---|
| throughput | 12.81 img/s | **41.93 img/s** |
| step time | 624 ms | **191 ms** |
| epoch (164k images) | 4h23m | **~1h05m** |
| device utilisation | 69.9% busy | **85.4% busy** |
| peak memory | 5.91 GiB | 6.49 GiB |

Everything below was measured, never assumed. Negative results are included because they are the expensive
part — and one of them overturned an earlier conclusion of this same report.

The complete working session — every command, every tool result, every wrong turn, in order — is published as
[`history.html`](https://tuan3w.github.io/rfdetr-training-throughput/history.html) (1,600 entries). It is
redacted by `export_session.py`, which drops known secret-bearing environment values, rewrites token shapes
(Atlassian, OpenRouter, Figma, generic 32-hex keys, `user:password@host` URLs) and maps internal hosts, paths
and names to neutral ones. The exporter re-scans its own output and refuses to write on a leak.

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
| `torch.compile` | +13.6% | Inductor fusion; −0.8 GiB. One-off ~11 min warm-up. |
| fused AdamW | +1.3% | Noise locally; 1.20× on the launch-bound A100. |
| **fused Triton matcher kernel** | **+20.0%** | Cost matrix in one launch, 174× standalone. |
| **exact GPU assignment solver** | **+20.9%** | SciPy's `rectangular_lsap` ported to CUDA. |
| **device-side unpacking** | **+3.1%** | 156 → 13 host/device syncs per step. |

### Kernel 1 — the Hungarian cost matrix (+20%)

The PyTorch profiler found one operator owning **15.3% of all device time in 5 calls per step**:
`aten::_cdist_forward`. rf-detr builds the matcher's cost matrix as a chain of separate kernels — class
gather, focal cost, `cdist` L1, two box conversions, a vmapped GIoU, then three multiplies and two adds —
and every pass reads and writes a full `[batch, 3900 queries, targets]` tensor. `torch.cdist(p=1)` on a
4-wide box dimension also lands on a generic kernel.

One Triton kernel now computes the whole thing in a single launch: **174× faster standalone** on the
production shape (25.4 ms → 0.145 ms). rf-detr ships no custom kernels at all, so this was unoptimised
ground.

### Kernel 2 — the solver, and a conclusion I got wrong twice

NVTX ranges showed the host spending **102 ms per microbatch inside the criterion** — as much as the entire
backward pass — because the SciPy assignment solve drains the GPU pipeline five times per step.

The instinctive fix was to move the solve to the GPU. rf-detr supports that (a Triton-backed batched solver),
and it **lost 35%**. The obvious reading was that ~520 *tiny* problems per step (5 decoder layers × 8 images
× 13 query groups, `[300 × T]`) are latency-bound and belong on the host, so an 8-thread SciPy pool was
accepted at **+13.3%**. An earlier version of this report said the solver "belongs on the CPU".

That was wrong. The −35% only disproved *that* solver, which is built for large square problems. Writing one
for this shape — a faithful CUDA port of SciPy's `rectangular_lsap`, one block per problem — gained a further
**+20.9%** (33.63 → 40.67 img/s) and eliminated the five `[3900 × T]` device-to-host copies entirely.
Criterion host time across the three generations: **102 → 72 → 42.8 ms** per microbatch.

Two details mattered:

- **Port SciPy, don't write a textbook Hungarian.** SciPy's inner loop removes exactly one column per
  iteration, which *bounds* the augmenting search. My first attempt used the usual formulation, had no such
  bound, and hung the GPU — a non-terminating block is a dead device, not a wrong answer.
- **Bound it by target count.** Cost is ~quadratic in targets (0.12 ms at 12, 1.40 ms at 108, 76.8 ms at
  300), and `_stack_padded` pads every problem to the batch's largest target count, so one crowded image
  drags all 520 problems into the square regime. But the host path scales the same way and loses at every
  size measured — 1.6 vs 48 ms at 128 targets, 7.0 vs 92 at 219, 76.8 vs 217 at 300 — so the bound sits past
  the dataset's worst image (219 targets) rather than where the kernel merely stops being fast.

Verification: **zero index differences** against SciPy on every matcher shape, objective gap `0.00e+00`.

### Then the syncs it exposed (+3.1%)

With the solve on the GPU, the indices were still copied to the host. `torch.cuda.set_sync_debug_mode`
attributed **156 synchronising operations per step**, ~130 of them to the criterion indexing CUDA tensors
with those host index tensors (`criterion.py:904`, `:906`, `:562`, `:722`). Unpacking the assignment on the
device leaves **13**. Padded columns are dropped with a cached selection index rather than a boolean mask,
since masked indexing has a data-dependent output size and would synchronise by itself. Pair sets are
identical across five target distributions.

## How the bottleneck was located

Each tool was escalated to only after the previous one was exhausted:

| tool | measurement | conclusion |
|---|---|---|
| `torch.profiler` | `aten::_cdist_forward` 37.6 ms/step, 15.3% of device time | → kernel 1 |
| Nsight Systems | device busy 69.9%; **25.7% of all idle in 104k gaps under 50 µs** | 3.8 µs launch latency × 4,835 launches/step |
| Nsight NVTX | criterion 102 ms vs backward 106 ms per microbatch | → kernel 2 |
| loader-only mode | ceiling 239 img/s vs ~34 consumed | 8× headroom: augmentation tuning cannot pay |
| `gap.py --cached-batches` | gap 41.6 ms with **zero loader workers** — unchanged | the inter-batch gap is not the loader |
| `sync_probe.py` | 156 syncs/step, ~130 from four criterion lines | → device-side unpacking |
| `hostgap.py` | `apply_func.py:134 to_item()` blocks **35.65 ms/microbatch** | Lightning waiting for the GPU, not host overhead |
| `tlparse` | 7 compile ids fail: `Expected IntList but got GenericList` | **`compile=True` never compiles the model** (below) |
| Nsight Systems, after | device busy **85.4%**, 48% of idle in 10,612 gaps of 5–50 µs | now genuinely GPU-bound |

Two of those rows killed entire branches of plausible-sounding work: the loader ceiling ruled out
augmentation/worker tuning, and `hostgap.py` showed the 42 ms inter-batch "gap" is mostly *overlapped* GPU
work — only ~3.5 ms of device idle is attributable to it, so there was no 42 ms of overhead to reclaim.

## An upstream bug: `compile=True` silently loses the backbone

Structured Dynamo logs show that with `compile=True` the model forward **never compiles at all**. Seven
compile ids, all rooted at `lwdetr.py:477`, fail with:

```
BackendCompilerFailed: isIntList() INTERNAL ASSERT FAILED ... Expected IntList but got GenericList
  from _upsample_bicubic2d_aa_backward(grad, [floordiv, floordiv], [1, 384, 36, 36], False)
```

The DINOv2 positional-embedding resize gets a **symbolic** output size because rf-detr compiles with
`dynamic=True`, and the antialiased-bicubic backward cannot accept one. `suppress_errors=True` then hides the
failure and the dominant compute runs eager — upstream names this exact subgraph at `module_model.py:439`.
So `torch.compile`'s +13.6% comes from everything *except* the model.

Three repairs were tried; all cost more than they return:

| repair | result |
|---|---|
| Disable Dynamo for just the resize | inductor then attempts the real graph: **22 min on one frame** without finishing, GPU idle, one core pinned on sympy shape reasoning over an ~800 KB FX graph (29 compile workers at ~2%) |
| `dynamic=False` (concrete shapes) | progresses at 2–3 min per resolution × **8 multi-scale sizes** (384…832) |
| Compile each DINOv2 block separately | **43.17 vs 45.24 img/s** — slower |

## What did not work

| attempt | result |
|---|---|
| rf-detr's GPU assignment solver | **−35%** — built for large square problems, not 520 tiny ones |
| Fused Triton deformable attention | +2.9% eager, **−4.0% under `compile`** |
| **Real Deformable-DETR CUDA kernel** (HF kernels hub) | **3.9–5.0× standalone, numerically equivalent, yet −2.4% end to end** (40.91 vs 41.93): it wants `[batch, sequence, heads, head_dim]` in fp32 while the caller hands it the transposed layout, and the permute-and-copy costs more than the faster backward saves |
| CUDA graphs | **OOM** — 12.3 GiB of private graph pools after 2 of 8 multi-scale shapes |
| Batch 12 / 16 (tested 3×) | −1% to −8.6% locally; **+12.7% on the A100** (108 SMs vs 24) |
| Pinned-memory host transfer (2 variants) | −16% to +2%, all noise |
| Fusing the 6 augmentation wrappers into one | −2% — hidden by loader headroom |
| `pack_targets=0` | −5.4% — the existing packing is a real win |
| More workers / prefetch, thread counts, EMA interval, expandable segments | all within noise |

## Correctness

A faster kernel that is subtly wrong would quietly damage training, so each was held to what a consumer
observes — the assignment decisions and the loss trajectory — not just to matching floats.

- Triton matcher vs upstream: ≤5e-06 across five production shapes; **3.8e-06 on real batches locally and
  exactly 0.0 on the A100**; degenerate zero-area boxes stay finite; non-L1 metrics fall back bit-exactly.
- **GPU solver vs SciPy: zero index differences**, objective gap `0.00e+00`, on every matcher shape including
  the square 520×300×300 case.
- **Device-side unpacking:** identical `(row, col)` pair sets across five target distributions; the 1e-06…5e-04
  cost deltas are fp32 summation order, not different matchings.
- **Loss trajectory**, deterministic loader: the kernel deviates 2.9e-02, versus **3.1e-02 between two
  identical unpatched runs** — smaller than the training loop's own nondeterminism.
- The benchmark entrypoint refuses to report a number unless every verifier passes while a patch is active.

## Recommended configuration

```
--sdpa-backend auto --fused-optimizer --no-gradient-checkpointing \
--batch-size 16 --grad-accum-steps 2
```

plus `compile=True` on the model constructor and the three accepted patches (`triton_matcher`, `cuda_lap`,
`device_indices`). Reduce `epochs` from 100 to 25–30 (rf-detr's docs recommend 20–30 above 10k images; this
dataset has 164k), and set `eval_interval 5`, `eval_batch_size 16`, `eval_max_dets 100`. Keep what was
already right: `pack_targets`, 16 workers, thread pinning, `expandable_segments:False`.

On the shared A100, the stack measured 10.04 → 16.70 img/s (**+66%**) even while time-slicing with the
co-tenant, before the solver work existed.

## Where the remaining time goes

The device is now **85.4% busy** (from 69.9%), so the step is genuinely GPU-bound and host-side work has
little left to win. Of the 14.6% idle, **48% is 10,612 gaps of 5–50 µs** — launch latency from an eager model
graph — and 25% is 20 gaps over 1 ms at microbatch boundaries. Removing the launch latency needs either CUDA
graphs (ruled out above) or a compiled model graph, which is exactly what the upstream `dynamic=True` bicubic
bug prevents. Device time itself is dominated by real work: bf16 cutlass GEMMs hold the top three entries at
138/125/75 ms per 1.4 s window.

## Upstream contribution

[roboflow/rf-detr#1486](https://github.com/roboflow/rf-detr/pull/1486) — replaces `torch.cdist(p=1)` in the
matcher with a bit-exact `pairwise_box_l1_cost`, the smallest dependency-free part of kernel 1. 1,169 tests
pass; the full CPU suite matches clean-`develop`'s failure set exactly.

## Reproducing

The harness is a small uv project: one file per concern — `cell.py` (measure one configuration),
`guard.py` (honesty invariants), `profile.py` / `nsys_gaps.py` / `nvtx.py` / `gap.py` / `sync_probe.py` /
`hostgap.py` (attribution), `kernels.py` / `kernels_lap.py` / `kernels_msda.py` / `kernels_msda_cuda.py`
(kernels, each with `--verify`/`--bench`), `patches.py` (opt-in monkeypatches, so the upstream checkout is
never modified), `verify_matcher.py` (real-batch equivalence), `report.py` / `export_report.py` (dashboards).

```bash
uv sync
python lab/make_subset.py <dataset> data/subset --train 3000
uv run python -m lab.kernels --verify --bench       # cost-matrix kernel
uv run python -m lab.kernels_lap --verify           # GPU solver vs SciPy
uv run python -m lab.cell --mode step --dataset-dir data/subset --steps 40
bash autoresearch.sh                                 # full gated benchmark
```

---

*Measured on an RTX 5060 Ti (development) and an A100-SXM4-40GB (cross-check), rf-detr 1.10.1,
torch 2.14.0+cu130, Triton 3.8.*
