# pack_gqa `crd2idx` rank mismatch with CUTLASS DSL >= 4.5

## Summary

`nvidia-cutlass-dsl==4.5.0.dev0` introduces stricter layout coalescing in CuTe. When a hierarchical layout like `(qhead_per_kvhead, seqlen_q):(stride_h, stride_m)` is contiguous (i.e., `stride_h * qhead_per_kvhead == stride_m`), CuTe 4.5 coalesces it into a flat layout `(N):(stride_h)`. The existing `pack_gqa.py` code then passes hierarchical coordinates `((h_idx, m_idx),)` to `crd2idx`, which rejects the rank mismatch.

## Error

```
error: unable to compute crd2idx with '!cute.layout<"(?):(1)">' and '!cute.coord<"((?,?))">'
```

at `pack_gqa.py` line 140 (and 3 other call sites with the same pattern).

## Root cause

`pack_gqa_layout()` creates a tensor with mode 0 = `(qhead_per_kvhead, seqlen_q):(head_stride, seqlen_stride)`. When this sub-layout is contiguous, CuTe 4.5 coalesces it to a single flat mode. The coordinate `((h_idx, m_idx),)` then has rank 2 but the layout has rank 1.

### When does coalescing happen?

The packed mode-0 strides are `(head_stride, seqlen_stride)` where `head_stride = T.stride[head_idx]` and `seqlen_stride = T.stride[0]`. Coalescing happens when `head_stride * qhead_per_kvhead == seqlen_stride`:

- **MHA** (`qhead_per_kvhead=1`): Always coalesces — size-1 mode is trivially contiguous.
- **MQA** (`qhead_per_kvhead=nheads`): Coalesces because `head_stride * nheads == seqlen_stride` for standard `(batch, seqlen, nheads, headdim)` layout.
- **GQA** (`1 < qhead_per_kvhead < nheads`): Does NOT coalesce because `head_stride * qhead_per_kvhead < seqlen_stride` (stride gap from non-adjacent heads).

### What triggers it?

All 4 call sites in `pack_gqa.py` that use `elem_pointer(tensor, ((h_idx, m_idx),))`:

1. `compute_ptr()` — called with `mQ[None, 0]` and `mO[None, 0]` (sliced tensors lose hierarchy)
2. `load_Q()` else branch — inline `elem_pointer(mQ_ptr_base, ((h_idx, m_idx),))`
3. `store_LSE()` else branch — `elem_pointer(mLSE, ((h_idx, m_idx),))`
4. `store_O()` else branch — inline `elem_pointer(mO_ptr_base, ((h_idx, m_idx),))`

The `[None, 0]` slicing (takes mode 0 at headdim=0) produces a tensor whose only mode is the packed `(qh, sq)` layout — which gets coalesced.

## What we tried

1. **Scalar flat index `crd2idx(idx, layout)`**: Compiles, but gives wrong pointer offsets. The coalesced layout `(N):(1)` has lost the physical stride, so `idx * 1 != h_idx * head_stride + m_idx * seqlen_stride`.

2. **Pass full 2D tensor with `((h_idx, m_idx), 0)` coord**: Also fails — CuTe 4.5 coalesces mode 0 even within the 2D tensor, producing layout `(?):(1)` and rejecting `((?,?),0)`.

3. **Constexpr branch `qhead_per_kvhead == 1`**: Only fixes MHA, not MQA.

## Likely fix directions

- **Reconstruct the layout explicitly**: After `[None, 0]` slicing, wrap the result in `cute.make_tensor(ptr, cute.make_layout((qh, sq), stride=(head_stride, seqlen_stride)))` using the known strides. This preserves the hierarchy.

- **Manual pointer arithmetic**: Skip `crd2idx` entirely. Compute `ptr + h_idx * head_stride + m_idx * seqlen_stride` directly. Requires passing the strides as parameters (they're available from the original tensor before slicing).

- **Prevent coalescing upstream**: Check if CuTe DSL 4.5 has an API to create "non-coalescable" layouts, or if `make_layout` has a flag to preserve hierarchy.

## Affected configurations

- MHA with `pack_gqa=True` or `pack_gqa=None` (MHA enables pack_gqa by default when num_splits > 1)
- MQA with any `pack_gqa` setting
- GQA is NOT affected (non-contiguous strides prevent coalescing)

## Reproduction

```bash
uv pip install --prerelease=allow 'nvidia-cutlass-dsl==4.5.0.dev0'
FLASH_ATTENTION_DISABLE_SPLIT=TRUE pytest tests/cute/test_flash_attn.py::test_flash_attn_output -x -k "64-False-0-0.0-False-False-False-mqa"
```

## Environment

- `nvidia-cutlass-dsl==4.5.0.dev0` (works on `4.4.2`)
- CUDA 13.0, PyTorch 2.11
- SM120 (RTX 5060 Ti), but issue is architecture-independent — affects SM80/SM90/SM100 pack_gqa paths too
