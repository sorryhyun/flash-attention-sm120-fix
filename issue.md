# `pack_gqa` broken on SM80/SM120 (non-TMA) forward paths: `crd2idx` rejects nested coordinates

## Summary

`PackGQA` methods in `pack_gqa.py` use `elem_pointer(tensor, ((h_idx, m_idx),))` which passes a nested coordinate to `crd2idx`. The MLIR IR layer rejects this because the layout type has been flattened to `(?):(?)` while the coordinate retains its nesting as `((?,?))`.

This affects the **SM80 base forward class** epilogue (`store_O`, `store_LSE`) and the non-TMA `load_Q` path. SM90/SM100 avoid this by using TMA for Q loading and their own TMA-based epilogues. SM120 (Blackwell GeForce) subclasses SM80 and hits the broken path.

**Not a cutlass-dsl version regression** — broken on 4.4.0, 4.4.1, 4.4.2, and 4.5.0.dev0 alike.

## Error

```
loc("...pack_gqa.py":140:20): error: unable to compute crd2idx
  with '!cute.layout<"(?):(?{i64 div=8})">' and '!cute.coord<"((?,?))">'
ValueError: Operation creation failed
```

## Root cause

`pack_gqa_layout()` creates a tensor with nested shape in mode 0:

```python
# pack_gqa.py:28-39
shape_packed = ((qhead_per_kvhead, T.shape[0]), ...)
stride_packed = ((head_stride, T.stride[0]), ...)
return cute.make_tensor(T.iterator, cute.make_layout(shape_packed, stride=stride_packed))
```

`compute_ptr()` then slices to a rank-1 view and indexes with a nested coordinate:

```python
# pack_gqa.py:140
tPrPtr[i] = utils.elem_pointer(tensor, ((h_idx, m_idx),)).toint()
```

The nested shape `(qhead_per_kvhead, seqlen_q)` is dynamic (both are `cutlass.Int32`, not `Constexpr`), so the MLIR layout type represents it as a single flat dynamic mode `(?):(?)`. The coordinate `((h_idx, m_idx),)` retains its hierarchical structure. `crd2idx` requires structural agreement between the two and rejects the mismatch.

## Why this was never caught

The `compute_ptr` → `elem_pointer` call exists in **all** `PackGQA` methods (including upstream). But:

1. **SM90 forward** (`flash_fwd_sm90.py`): Uses TMA for Q loading (`use_tma_Q=True`) and has its own TMA-based O/LSE epilogue. `PackGQA.load_Q`/`store_O`/`store_LSE` are **never called**, so `compute_ptr` is **never JIT-compiled**. The only exception is when `pack_gqa and tile_m % qhead_per_kvhead != 0`, which disables TMA for Q and would hit this bug.

2. **SM100 forward** (`flash_fwd_sm100.py`): Also TMA-based, same situation.

3. **SM80 forward** (`flash_fwd.py`): The base class epilogue calls `pack_gqa.store_O` and `pack_gqa.store_LSE` which call `compute_ptr`. This is instantiated when `arch // 10 == 8` in `interface.py`. **This path is broken on any cutlass-dsl version**, but SM80 GPUs are rare in CI (tests typically run on H100/B200).

4. **SM120 forward** (`flash_fwd_sm120.py`): Subclasses SM80, inherits the non-TMA epilogue. Always hits the broken `compute_ptr` path.

## Affected call sites in `pack_gqa.py`

| Line | Method | Callers |
|------|--------|---------|
| 140 | `compute_ptr()` | Called by `load_Q`, `store_LSE`, `store_O` below |
| 162/200 | `load_Q()` | SM80 fwd (line 788 in sm90 non-TMA-Q path), SM120 fwd |
| 203/270 | `store_LSE()` | SM80 fwd epilogue (line 383), SM120 fwd |
| 239/334 | `store_O()` | SM80 fwd epilogue (line 442), SM120 fwd |

(Our fork added `else` branches for `WARP_SIZE % threads_per_row != 0` at lines 200, 270, 334. Upstream only has the assert-guarded main path, but `compute_ptr` at line 140 is shared.)

## Reproduction

```python
import torch
from flash_attn.cute import flash_attn_func

# On SM80 or SM120 GPU:
q = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device='cuda')
out, lse = flash_attn_func(q, k, v, pack_gqa=True)
# ValueError: Operation creation failed
```

Fails for MHA (`qhead_per_kvhead=1`), GQA, and MQA alike.

## Possible fix

The simplest fix is manual pointer arithmetic in `compute_ptr`, bypassing `crd2idx`:

```python
# Instead of:
tPrPtr[i] = utils.elem_pointer(tensor, ((h_idx, m_idx),)).toint()

# Use:
base_ptr = tensor.iterator.toint()
head_stride = tensor.stride[0][0]   # stride for qhead_per_kvhead dim
seq_stride = tensor.stride[0][1]    # stride for seqlen dim
tPrPtr[i] = base_ptr + h_idx * head_stride + m_idx * seq_stride
```

This avoids the `crd2idx` type mismatch entirely by computing the offset arithmetically. The strides are available from the tensor before the `[None, 0]` slicing collapses the type info.

Alternatively, this could be fixed in the CUTLASS DSL IR by allowing `crd2idx` to accept nested coordinates against flat dynamic layout types (matching the C++ CuTe `crd2idx` behavior).

## Current workaround

`interface.py` gates `pack_gqa` on cutlass-dsl version, but since this is version-independent, the correct guard is architecture-based:

```python
# Disable pack_gqa on SM80-based forward paths (SM80, SM120)
if pack_gqa and compute_capability in [8, 12]:
    pack_gqa = False
```

## Environment

- `nvidia-cutlass-dsl` 4.4.0 / 4.4.1 / 4.4.2 / 4.5.0.dev0 — all affected
- PyTorch 2.11.0+cu130, CUDA 13.0
- Tested on SM120 (RTX 5060 Ti). SM80 (A100) would also be affected.
- SM90 (H100) and SM100 (B200) are **not** affected because TMA paths bypass `PackGQA` methods.
