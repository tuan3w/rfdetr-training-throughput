#!/usr/bin/env python3
"""Exact batched linear-assignment solver on the GPU, sized for RF-DETR's matcher.


Why not the existing options
----------------------------
The matcher needs ~520 assignment problems solved per training step (5 decoder
layers x 8 images x 13 query groups), each ``[300 queries x T targets]`` with T
typically 1-60. Two solvers already exist and both are wrong for this shape:

* **SciPy on the host** (what RF-DETR falls back to, and what the crashy Triton
  path is usually replaced with) forces the whole cost matrix to the CPU. Nsight
  shows 11-13 ms/step of pageable device-to-host copy plus a blocking
  ``cudaStreamSynchronize``, and NVTX attributes ~72 ms of host time per
  microbatch to the criterion that waits on it. Copying five ``[3900 x T]``
  matrices off the device every step to solve a problem the device could solve
  is the wrong architecture, however fast the CPU solve itself is.
* **``torch_linear_assignment``'s Triton path** is built for large, roughly
  square problems; measured end to end on this workload it cost 35% throughput
  because its per-call overhead (~10 ms) dwarfs these tiny problems.

What this does
--------------
One CUDA block per problem, solving it with Jonker-Volgenant shortest augmenting
paths — the same algorithm SciPy uses, so the result is the exact optimum, not an
auction/epsilon approximation. Each block keeps the duals and the alternating
tree in shared memory and uses its threads to parallelise the O(W) column scan
that dominates each augmentation; the 520 problems run concurrently across SMs.
Nothing leaves the device except the final small index tensors the matcher
already transfers.

Scope: CUDA float32 costs with ``targets <= queries`` and ``queries`` within the
shared-memory budget. Everything else falls back to the caller's original solver,
so CPU/MPS, oversized problems and the ``targets > queries`` orientation behave
exactly as before.

    uv run python -m lab.kernels_lap --verify --bench
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

#: Largest ``tasks`` extent the kernel accepts.
#:
#: Cost per problem is roughly quadratic in tasks (520 problems of 300 queries:
#: 0.12 ms at 12 tasks, 1.40 ms at 108, 7.0 ms at 219, 76.8 ms at 300), so this
#: bound existed first to stop a single crowded image -- `_stack_padded` pads
#: every problem to the batch's largest target count -- from dragging the whole
#: batch into the square regime.
#:
#: Measured against the host path it replaces (device-to-host copy plus SciPy on
#: eight threads) the device solver is still far ahead at every size: 1.6 vs 48 ms
#: at 128 tasks, 7.0 vs 92 ms at 219, 76.8 vs 217 ms at 300. So the bound is set
#: past the dataset's worst image (219 targets) rather than at the point where the
#: kernel merely stops being fast, which leaves no batch on the host path: at
#: batch 8 a bound of 128 fell back on 2.5% of batches and at batch 16 on 3.7%.
MAX_TASKS = 256

#: Largest ``queries`` extent the kernel accepts. The block keeps five per-column
#: arrays in shared memory (17 bytes per column), so 2048 columns is ~35 KiB —
#: inside the 48 KiB default limit with room for the reduction scratch.
MAX_QUERIES = 2048

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

constexpr float kInfinity = 3.402823466e+38f;

// One block per problem. This is a direct port of SciPy's
// scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (Crouse's shortest
// augmenting path), chosen over the textbook e-maxx formulation for one
// property that matters on a GPU: its inner loop removes exactly one column
// from `remaining` per iteration, so an augmentation provably terminates in at
// most `num_cols` steps. A block that cannot terminate is a hung kernel, not a
// wrong answer, so the bound is the whole point.
//
// Rows are tasks (targets) and columns are workers (queries), which satisfies
// SciPy's requirement that rows <= columns for the untransposed orientation.
__global__ void lsap_kernel(
    const float* __restrict__ cost,   // [problems, num_cols, num_rows] (worker-major)
    long* __restrict__ matching,      // [problems, num_cols]
    const int num_cols,               // workers / queries
    const int num_rows) {             // tasks / targets
  const int problem = blockIdx.x;
  const int threads = blockDim.x;
  const int tid = threadIdx.x;

  extern __shared__ char shared_raw[];
  char* cursor = shared_raw;
  float* dual_row = reinterpret_cast<float*>(cursor);            cursor += num_rows * sizeof(float);
  float* dual_col = reinterpret_cast<float*>(cursor);            cursor += num_cols * sizeof(float);
  float* shortest = reinterpret_cast<float*>(cursor);            cursor += num_cols * sizeof(float);
  int* path = reinterpret_cast<int*>(cursor);                    cursor += num_cols * sizeof(int);
  int* col_of_row = reinterpret_cast<int*>(cursor);              cursor += num_rows * sizeof(int);
  int* row_of_col = reinterpret_cast<int*>(cursor);              cursor += num_cols * sizeof(int);
  int* remaining = reinterpret_cast<int*>(cursor);               cursor += num_cols * sizeof(int);
  float* reduce_cost = reinterpret_cast<float*>(cursor);         cursor += threads * sizeof(float);
  int* reduce_slot = reinterpret_cast<int*>(cursor);             cursor += threads * sizeof(int);
  int* reduce_free = reinterpret_cast<int*>(cursor);             cursor += threads * sizeof(int);
  int* scalars = reinterpret_cast<int*>(cursor);                 cursor += 4 * sizeof(int);
  float* min_value = reinterpret_cast<float*>(cursor);           cursor += sizeof(float);
  char* in_rows = reinterpret_cast<char*>(cursor);               cursor += num_rows * sizeof(char);
  char* in_cols = reinterpret_cast<char*>(cursor);

  // scalars[0] = current row, [1] = sink, [2] = remaining count, [3] = chosen slot
  const float* problem_cost = cost + static_cast<long>(problem) * num_cols * num_rows;

  for (int row = tid; row < num_rows; row += threads) {
    dual_row[row] = 0.0f;
    col_of_row[row] = -1;
  }
  for (int col = tid; col < num_cols; col += threads) {
    dual_col[col] = 0.0f;
    row_of_col[col] = -1;
  }
  __syncthreads();

  for (int current_row = 0; current_row < num_rows; ++current_row) {
    for (int col = tid; col < num_cols; col += threads) {
      // Reverse fill matches SciPy's comment on gh-11602: it makes a constant
      // cost matrix resolve to the identity assignment.
      remaining[col] = num_cols - col - 1;
      shortest[col] = kInfinity;
      in_cols[col] = 0;
      path[col] = -1;
    }
    for (int row = tid; row < num_rows; row += threads) {
      in_rows[row] = 0;
    }
    if (tid == 0) {
      scalars[0] = current_row;
      scalars[1] = -1;
      scalars[2] = num_cols;
      min_value[0] = 0.0f;
    }
    __syncthreads();

    // At most one column leaves `remaining` per iteration.
    for (int iteration = 0; iteration < num_cols; ++iteration) {
      if (scalars[1] >= 0) {
        break;
      }
      const int row = scalars[0];
      const int num_remaining = scalars[2];
      const float accumulated = min_value[0];
      if (tid == 0) {
        in_rows[row] = 1;
      }
      __syncthreads();

      float best_cost = kInfinity;
      int best_slot = -1;
      int best_free = 0;
      for (int slot = tid; slot < num_remaining; slot += threads) {
        const int col = remaining[slot];
        const float reduced = accumulated +
            problem_cost[static_cast<long>(col) * num_rows + row] - dual_row[row] - dual_col[col];
        if (reduced < shortest[col]) {
          path[col] = row;
          shortest[col] = reduced;
        }
        const int is_free = row_of_col[col] == -1 ? 1 : 0;
        // SciPy prefers, among equal costs, a column that yields a new sink.
        if (shortest[col] < best_cost || (shortest[col] == best_cost && is_free && !best_free)) {
          best_cost = shortest[col];
          best_slot = slot;
          best_free = is_free;
        }
      }
      reduce_cost[tid] = best_cost;
      reduce_slot[tid] = best_slot;
      reduce_free[tid] = best_free;
      __syncthreads();

      for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          const int other = tid + stride;
          const bool take = reduce_slot[other] >= 0 &&
              (reduce_slot[tid] < 0 || reduce_cost[other] < reduce_cost[tid] ||
               (reduce_cost[other] == reduce_cost[tid] && reduce_free[other] && !reduce_free[tid]));
          if (take) {
            reduce_cost[tid] = reduce_cost[other];
            reduce_slot[tid] = reduce_slot[other];
            reduce_free[tid] = reduce_free[other];
          }
        }
        __syncthreads();
      }

      if (tid == 0) {
        const float lowest = reduce_cost[0];
        const int slot = reduce_slot[0];
        if (slot < 0 || lowest == kInfinity) {
          // Infeasible problem: leave the sink unset and stop.
          scalars[2] = 0;
          scalars[1] = -2;
        } else {
          min_value[0] = lowest;
          const int col = remaining[slot];
          if (row_of_col[col] == -1) {
            scalars[1] = col;
          } else {
            scalars[0] = row_of_col[col];
          }
          in_cols[col] = 1;
          const int last = scalars[2] - 1;
          remaining[slot] = remaining[last];
          scalars[2] = last;
        }
      }
      __syncthreads();
    }

    const int sink = scalars[1];
    if (sink < 0) {
      // Infeasible: this row stays unassigned, like SciPy returning an error.
      continue;
    }

    // Dual update, exactly SciPy's: rows in the tree shift by the residual
    // between the path cost that reached them and the final minimum.
    const float minimum = min_value[0];
    if (tid == 0) {
      dual_row[current_row] += minimum;
    }
    for (int row = tid; row < num_rows; row += threads) {
      if (in_rows[row] && row != current_row) {
        dual_row[row] += minimum - shortest[col_of_row[row]];
      }
    }
    for (int col = tid; col < num_cols; col += threads) {
      if (in_cols[col]) {
        dual_col[col] -= minimum - shortest[col];
      }
    }
    __syncthreads();

    if (tid == 0) {
      int col = sink;
      while (true) {
        const int row = path[col];
        row_of_col[col] = row;
        const int previous = col_of_row[row];
        col_of_row[row] = col;
        col = previous;
        if (row == current_row) {
          break;
        }
      }
    }
    __syncthreads();
  }

  long* problem_matching = matching + static_cast<long>(problem) * num_cols;
  for (int col = tid; col < num_cols; col += threads) {
    problem_matching[col] = static_cast<long>(row_of_col[col]);
  }
}

}  // namespace

torch::Tensor batch_lap_cuda(torch::Tensor cost) {
  TORCH_CHECK(cost.is_cuda(), "cost must be a CUDA tensor");
  TORCH_CHECK(cost.dim() == 3, "cost must be [problems, workers, tasks]");
  TORCH_CHECK(cost.scalar_type() == torch::kFloat32, "cost must be float32");
  const at::cuda::OptionalCUDAGuard guard(device_of(cost));

  auto contiguous = cost.contiguous();
  const int problems = static_cast<int>(contiguous.size(0));
  const int workers = static_cast<int>(contiguous.size(1));
  const int tasks = static_cast<int>(contiguous.size(2));
  TORCH_CHECK(tasks <= workers, "this kernel requires tasks <= workers");

  auto matching = torch::empty({problems, workers}, contiguous.options().dtype(torch::kLong));
  if (problems == 0 || workers == 0) {
    return matching;
  }
  if (tasks == 0) {
    matching.fill_(-1);
    return matching;
  }

  const int threads = 256;
  // Per task: dual_row (float), col_of_row (int), in_rows (char).
  // Per worker: dual_col + shortest (floats), path + row_of_col + remaining
  // (ints), in_cols (char). Undersizing this by even one array overruns into the
  // next block's shared memory, which surfaces as an illegal access.
  const size_t shared_bytes =
      tasks * (sizeof(float) + sizeof(int) + sizeof(char)) +
      workers * (2 * sizeof(float) + 3 * sizeof(int) + sizeof(char)) +
      threads * (sizeof(float) + 2 * sizeof(int)) + 4 * sizeof(int) + sizeof(float);

  lsap_kernel<<<problems, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      contiguous.data_ptr<float>(), matching.data_ptr<long>(), workers, tasks);
  return matching;
}
"""

#: ``load_inline`` writes its own pybind module, so the CUDA translation unit must
#: only define the entry point and the C++ side must declare it.
_CPP_SOURCE = "torch::Tensor batch_lap_cuda(torch::Tensor cost);"

_extension: Any = None


def extension() -> Any:
    """Compile (once) and return the CUDA extension."""
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load_inline

        _extension = load_inline(
            name="rfdetr_lab_batch_lap",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["batch_lap_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _extension


def batch_lap(cost: torch.Tensor) -> torch.Tensor:
    """Solve a batch of assignment problems on the device.

    Args:
        cost: ``[problems, workers, tasks]`` float32 CUDA tensor.

    Returns:
        ``[problems, workers]`` int64 tensor giving each worker's task, ``-1`` when
        the worker is unassigned — the contract of
        ``torch_linear_assignment.batch_linear_assignment``.

    Raises:
        ValueError: If the batch is outside the supported envelope. Callers should
            test :func:`supported` first and keep their own fallback for the rest;
            the block's shared-memory footprint grows with ``workers``, so an
            oversized problem would fail at launch with a bare CUDA error.
    """
    if not supported(cost):
        raise ValueError(
            "unsupported batch for the device solver: expected a float32 CUDA tensor of shape "
            f"[problems, workers, tasks] with tasks <= workers <= {MAX_QUERIES}, got "
            f"{tuple(cost.shape)} ({cost.dtype}, cuda={cost.is_cuda})"
        )
    return extension().batch_lap_cuda(cost)


def supported(cost: torch.Tensor) -> bool:
    """Whether this solver can take a given batch."""
    return (
        cost.is_cuda
        and cost.dim() == 3
        and cost.dtype is torch.float32
        and cost.shape[2] <= cost.shape[1]
        and cost.shape[1] <= MAX_QUERIES
        and cost.shape[2] <= MAX_TASKS
    )


def reference_matching(cost: torch.Tensor) -> torch.Tensor:
    """SciPy's optimum, in the same layout, for verification."""
    from scipy.optimize import linear_sum_assignment

    host = cost.detach().cpu().numpy()
    matching = torch.full((cost.shape[0], cost.shape[1]), -1, dtype=torch.long)
    for index in range(cost.shape[0]):
        rows, cols = linear_sum_assignment(host[index])
        matching[index, torch.from_numpy(rows)] = torch.from_numpy(cols).to(torch.long)
    return matching


def assigned_cost(cost: torch.Tensor, matching: torch.Tensor) -> torch.Tensor:
    """Total assigned cost per problem, for comparing two optimal solutions."""
    mask = matching >= 0
    safe = matching.clamp(min=0).unsqueeze(-1)
    picked = cost.gather(2, safe).squeeze(-1)
    return (picked * mask).sum(dim=1)


#: Problem shapes the matcher actually produces: 5 layers x 8 images x 13 groups
#: of [300 queries x T targets], plus edge cases.
SHAPES: tuple[tuple[int, int, int], ...] = (
    (520, 300, 1),
    (520, 300, 12),
    (520, 300, 37),
    (104, 300, 60),
    (40, 300, 300),
    (1, 16, 16),
    (3, 64, 0),
)


def verify() -> int:
    """Check optimality and validity against SciPy on the matcher's shapes."""
    failures = 0
    generator = torch.Generator(device="cuda").manual_seed(4)
    for problems, workers, tasks in SHAPES:
        cost = torch.rand((problems, workers, tasks), device="cuda", generator=generator) * 10 - 5
        if not supported(cost):
            print(f"-- ({problems},{workers},{tasks}): outside the envelope, caller falls back")
            continue
        if tasks == 0:
            matching = batch_lap(cost)
            status = "ok" if bool((matching == -1).all()) else "FAIL"
            failures += status == "FAIL"
            print(f"{status} ({problems},{workers},{tasks}): all workers unassigned")
            continue

        matching = batch_lap(cost)
        expected = reference_matching(cost)

        # Validity: every task assigned exactly once, no duplicate workers.
        assigned = matching[matching >= 0]
        counts = torch.bincount(assigned.view(-1), minlength=tasks)
        valid = bool((matching >= 0).sum() == problems * tasks) and bool((counts == problems).all())

        gap = (assigned_cost(cost, matching) - assigned_cost(cost, expected.cuda())).abs().max().item()
        exact = gap <= 1e-4 * max(1.0, cost.abs().max().item())
        identical = int((matching != expected.cuda()).sum())

        status = "ok" if valid and exact else "FAIL"
        failures += status == "FAIL"
        print(
            f"{status} ({problems},{workers},{tasks}): valid={valid} objective gap {gap:.2e} "
            f"index differences {identical}"
        )
    return 1 if failures else 0


def bench() -> int:
    """Time the device solver against the host SciPy path it replaces."""
    import time

    from scipy.optimize import linear_sum_assignment

    generator = torch.Generator(device="cuda").manual_seed(5)
    print(f"{'shape':22s} {'scipy+D2H ms':>13s} {'gpu ms':>9s} {'speedup':>8s}")
    for problems, workers, tasks in ((520, 300, 12), (520, 300, 37), (104, 300, 60), (40, 300, 300)):
        cost = torch.rand((problems, workers, tasks), device="cuda", generator=generator)

        def host_solve() -> None:
            host = cost.cpu().numpy()
            for index in range(problems):
                linear_sum_assignment(host[index])

        def device_solve() -> None:
            batch_lap(cost)
            torch.cuda.synchronize()

        for _ in range(2):
            host_solve()
            device_solve()
        start = time.perf_counter()
        for _ in range(5):
            host_solve()
        host_ms = (time.perf_counter() - start) / 5 * 1e3
        start = time.perf_counter()
        for _ in range(5):
            device_solve()
        device_ms = (time.perf_counter() - start) / 5 * 1e3
        label = f"({problems},{workers},{tasks})"
        print(f"{label:22s} {host_ms:13.2f} {device_ms:9.2f} {host_ms / device_ms:7.2f}x")
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
