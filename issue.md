# Flash Attention 4: SM120 (Blackwell GeForce) Bugs

## Environment

- GPU: NVIDIA GeForce RTX 5060 Ti (SM 12.0)
- CUDA: 13.0
- PyTorch: 2.11.0+cu130
- flash-attn-4: 4.0.0b6.dev10 (original) / sisgrad fork `dz/sm120_tma_optimized`
- nvidia-cutlass-dsl: 4.4.2
- quack-kernels: 0.3.7

## Bug 1: Forward pass — TMA O-store crash (`flash_fwd.py`)

### Symptom

```
File "flash_attn/cute/flash_fwd.py", line 399, in epilogue
    store_O, _, _ = copy_utils.tma_get_copy_fn(
        tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
    )
File "quack/copy_utils.py", line 775, in tma_get_copy_fn
    s, g = cpasync.tma_partition(atom, ...)
File "nvidia_cutlass_dsl/.../helpers.py", line 209, in tma_partition
    atom._trait.value,
AttributeError: 'NoneType' object has no attribute '_trait'
```

### Root cause

`flash_fwd.py` line 652:

```python
self.use_tma_O = self.arch >= Arch.sm_90
```

SM120 is >= SM90, so `use_tma_O = True`. But SM120 uses SM80 MMA instructions and the CuTe TMA descriptor for the O epilogue is not initialized for this architecture — `atom._trait` is `None`.

Note: `FlashAttentionForwardSm120` sets `arch = 80` as a class attribute, but the parent `__init__` (`FlashAttentionForwardSm80.__init__`) overwrites it at line 110:

```python
self.arch = BaseDSL._get_dsl().get_arch_enum()  # returns SM 12.0 at runtime
```

So the class-level `arch = 80` override in `flash_fwd_sm120.py` has no effect on `use_tma_O`.

### Fix

```python
# flash_fwd.py line 652
self.use_tma_O = self.arch >= Arch.sm_90 and self.arch < Arch.sm_120
```

## Bug 2: Backward pass — `dQ_single_wg` unbound (`interface.py`)

### Symptom

```
File "flash_attn/cute/interface.py", line 1321, in _flash_attn_bwd
    dQ_single_wg,
UnboundLocalError: cannot access local variable 'dQ_single_wg' where it is not associated with a value
```

### Root cause

In `_flash_attn_bwd`, the SM120 config block (lines 1005-1028) sets tile sizes, swap flags, atom layouts, and `V_in_regs`, but does not set `dQ_single_wg`. The compile_key tuple for `arch // 10 in [8, 9, 12]` (line 1297) references `dQ_single_wg`, causing `UnboundLocalError`.

### Fix

Add `dQ_single_wg = False` to the SM120 block (consistent with SM80 backward which uses a single warp group):

```python
# interface.py, after line 1028
        dQ_single_wg = False
```

## Bug 3 (API change): `flash_attn_func` now returns `(out, lse)` tuple

The updated `flash_attn_func` / `FlashAttnFunc.apply` always returns `(out, lse)` regardless of `return_lse`. Callers expecting a single tensor will get `AttributeError: 'tuple' object has no attribute 'reshape'`.

This is not strictly a bug in flash-attn-4 but a breaking API change from the previous version where only the output tensor was returned.

## Fix commit

https://github.com/sorryhyun/flash-attention-sm120-fix/tree/dz/sm120_tma_optimized
