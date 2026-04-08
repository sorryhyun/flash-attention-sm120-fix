# Green Context SM Partitioning

CUDA green context utility for running independent kernels on disjoint SM subsets.

## What green context does

Green contexts (CUDA 12.0+) partition a GPU's SMs into disjoint sets, each with its own CUDA stream. Kernels launched on a partition stream are restricted to that partition's SMs by hardware.

This is useful when you have **genuinely different workloads** that can overlap — for example, running a small auxiliary model while the main model runs, or overlapping gradient reduction with optimizer steps in training.

## What it does NOT help

**CFG dual-pass is NOT a good use case.** Both positive and negative guidance passes are compute-bound and identical. Splitting SMs in half means each pass runs at ~half throughput. Concurrent execution on disjoint partitions just recovers what was lost — net zero benefit, plus synchronization overhead. Benchmarked result: **~2x slower** on RTX 5060 Ti (1.87 s/step vs 1.10 s/step).

For faster CFG, use **batched CFG** instead — concatenate positive/negative inputs along the batch dimension and run a single forward pass with batch_size=2.

## GreenContext API

```python
from flash_attn.cute.green_context import GreenContext

# As context manager
with GreenContext(num_sms=16) as ctx:
    with torch.cuda.stream(ctx.torch_stream):
        # kernels here run on 16 SMs
        result_a = model_a(x)
    with torch.cuda.stream(ctx.remainder_torch_stream):
        # kernels here run on the remaining SMs (concurrent)
        result_b = model_b(y)

# Manual lifecycle
ctx = GreenContext(num_sms=16)
ctx.create()
stream = ctx.stream           # CUstream for direct kernel launch
sm_count = ctx.sm_count       # actual partition SM count (aligned)
ctx.destroy()
```

### SM alignment

On CUDA 13.0+, requested SM count is rounded down to `smCoscheduledAlignment` (e.g., alignment=8 on RTX 5060 Ti: requesting 18 SMs → 16 actual). A warning is emitted when alignment changes the count.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `stream` | `CUstream` | Primary partition raw CUDA stream |
| `remainder_stream` | `CUstream` | Remainder partition raw CUDA stream |
| `sm_count` | `int` | Primary partition actual SM count |
| `remainder_sm_count` | `int` | Remainder SM count |
| `total_sms` | `int` | Full device SM count |
| `torch_stream` | `torch.cuda.Stream` | Primary as PyTorch ExternalStream |
| `remainder_torch_stream` | `torch.cuda.Stream` | Remainder as PyTorch ExternalStream |

## When green context helps

- **Heterogeneous workloads**: one partition runs a memory-bound kernel while the other runs a compute-bound kernel
- **Auxiliary models**: small classifier/router model overlapping with main model forward
- **Training pipeline**: gradient all_reduce on one partition while optimizer runs on another
- **Persistent FA4 kernels**: `StaticPersistentTileScheduler` grid sizing respects partition SM count via `sm_count` parameter

## When it does NOT help

- **Identical workloads** (like CFG dual-pass): splitting SMs just halves throughput per workload
- **Already GPU-saturating** kernels: no idle SMs to fill with concurrent work
- **Sequential dependencies**: if workload B depends on workload A's output

## Files

| File | Purpose |
|------|---------|
| `flash_attn/cute/green_context.py` | `GreenContext` utility class |
| `flash_attn/cute/tile_scheduler.py` | `StaticPersistentTileScheduler.get_grid_shape()` accepts `sm_count` override |
| `flash_attn/cute/__init__.py` | Exports `GreenContext` |
| `inference.py` | `--green_context` / `--green_context_sms` CLI flags (for experimentation) |
| `library/inference_pipeline.py` | Concurrent CFG path (disabled by default) |

## Limitations

- Requires CUDA driver >= 12.0
- CUDA 13.0+ recommended for proper SM co-scheduling alignment
- Concurrent execution doubles peak single-block activation memory (~200-400MB)
- Model must be safe for concurrent read-only inference (no shared mutable state)
