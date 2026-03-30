# SM120 Flash Attention 4: nan loss investigation

## Symptom

Training starts but loss immediately shows `nan`:
```
steps:   1%|▌  | 10/744 [00:30<36:53, 3.02s/it, avr_loss=nan]
```

Forward and backward passes complete without errors (after Bug 1 & 2 fixes).

## Suspected causes

### 1. SM120 backward kernel producing wrong gradients

The SM120 backward path uses `FlashAttentionBackwardSm120` (subclass of `FlashAttentionBackwardSm80`). There may be additional uninitialized variables or wrong config values similar to the `dQ_single_wg` bug. The backward could silently produce garbage gradients that propagate as nan.

**How to test:** Run with `attn_mode = "flash"` (FA2) and confirm loss is finite. If so, the issue is in FA4's SM120 backward.

### 2. `use_tma_O` fix — wrong fallback path in backward

Our fix `self.use_tma_O = self.arch >= Arch.sm_90 and self.arch < Arch.sm_120` only patches the **forward** kernel. The backward kernel (`flash_bwd.py` / `flash_bwd_sm120.py`) may have a similar TMA path issue that silently corrupts output instead of crashing.

**How to test:** Check `flash_bwd.py` and `flash_bwd_sm120.py` for similar `self.arch >= Arch.sm_90` guards.

### 3. `softmax_scale` not passed correctly

The new `flash_attn_func` signature may have changed parameter ordering. If `softmax_scale` is being interpreted as a different argument, attention scores could overflow.

**How to test:** Add a print in the wrapper to verify kwargs are passed correctly.

### 4. Output tuple unpacking discards autograd graph

Our wrapper does:
```python
def flash_attn_4_func(*args, **kwargs):
    out, _lse = _flash_attn_4_func_raw(*args, **kwargs)
    return out
```

If `FlashAttnFunc.apply` returns a tuple where both elements participate in the autograd graph, discarding `_lse` might not cause nan directly, but could affect gradient computation if the backward function expects `dlse` to be non-None.

**How to test:** Check if `FlashAttnFunc.backward` handles `dlse=None` gracefully.

### 5. SM120 kernel numerical precision issue

The SM120 kernel uses SM80 MMA (mma.sync.aligned.m16n8k16) with different shared memory capacity (99 KB). Tile sizes or accumulation order differences could cause numerical issues with certain head dimensions or sequence lengths.

**How to test:** Run a small standalone test comparing FA4 SM120 output vs FA2 output on the same input.

## Root cause: Bug 4 — `StMatrix8x8x16bOp` used with SM80 MMA on SM120

`utils.get_smem_store_atom` (utils.py:222) selects `StMatrix8x8x16bOp` (`stmatrix.sync.aligned`) for `arch >= 90`. SM120 passes this check but uses SM80 `mma.sync` MMA, whose output register layout is incompatible with `stmatrix`'s thread-to-shared-memory mapping. Result: corrupted O in forward, corrupted dQ in backward.

**Affected call sites:**
1. `flash_fwd.py:347` — forward epilogue, storing O registers → smem
2. `flash_bwd_postprocess.py:537` — backward postprocess, storing dQ registers → smem

**Fix:** Exclude SM120 from the stmatrix path:
```python
# utils.py line 222
if const_expr(arch < 90 or arch >= 120 or element_type.width != 16):
```

**Verify:** `python test_sm120_nan.py`

## Priority

Test #1 first — if `attn_mode = "flash"` produces valid loss, it confirms the issue is FA4 SM120-specific and narrows the search.
