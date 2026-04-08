# Copyright (c) 2025, Seunghyun Ji.
# CUDA Green Context utility for SM partitioning in Flash Attention kernels.

"""CUDA Green Context utility for SM partitioning.

Enables co-scheduling independent kernels on disjoint SM subsets for improved
GPU utilization. For example, running FA4 attention on 16 SMs while DiT FFN
runs on the remaining 22 SMs concurrently.

Usage::

    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.green_context import GreenContext

    # As context manager — simplest usage
    with GreenContext(num_sms=16) as ctx:
        with torch.cuda.stream(ctx.torch_stream):
            out, lse = flash_attn_func(q, k, v, causal=True)

    # Two partitions for concurrent kernels
    with GreenContext(num_sms=16) as ctx:
        # Launch attention on primary partition
        with torch.cuda.stream(ctx.torch_stream):
            out, lse = flash_attn_func(q, k, v)
        # Launch FFN on remainder partition (concurrently)
        with torch.cuda.stream(ctx.remainder_torch_stream):
            ffn_out = ffn(x)

    # Manual lifecycle for long-lived partitions
    ctx = GreenContext(num_sms=16)
    ctx.create()
    # ... use ctx.stream, ctx.sm_count ...
    ctx.destroy()

Requires CUDA driver version >= 12.0. On CUDA 13.0+, the requested SM count
is aligned down to ``smCoscheduledAlignment`` for proper cluster scheduling.
"""

from __future__ import annotations

import warnings
from typing import Optional

import cuda.bindings.driver as drv

try:
    import torch
except ImportError:
    torch = None


def _check(result):
    """Unwrap CUDA driver API result tuple, raising on error."""
    err = result[0]
    if err.value != 0:
        try:
            _, name = drv.cuGetErrorName(err)
        except Exception:
            name = f"error code {err.value}"
        raise RuntimeError(f"CUDA driver error: {name}")
    if len(result) == 2:
        return result[1]
    if len(result) > 2:
        return result[1:]
    return None


class GreenContext:
    """CUDA green context for SM partitioning.

    Creates two disjoint SM partitions on the GPU: a *primary* partition with
    the requested number of SMs and a *remainder* partition with the rest.
    Each partition gets its own CUDA stream that restricts kernel execution
    to that partition's SMs.

    Parameters
    ----------
    num_sms : int
        Requested number of SMs for the primary partition. On CUDA 13.0+,
        this is aligned down to ``smCoscheduledAlignment``. A warning is
        emitted if the actual count differs from the request.
    device_id : int
        CUDA device ordinal. Default: 0.
    """

    def __init__(self, num_sms: int, device_id: int = 0):
        # Set lifecycle flags first so __del__ is safe even if __init__ raises
        self._created = False
        self._destroyed = False

        if num_sms <= 0:
            raise ValueError(f"num_sms must be positive, got {num_sms}")
        self._requested_sms = num_sms
        self._device_id = device_id

        # Handles — populated by create()
        self._device: Optional[drv.CUdevice] = None
        self._primary_ctx: Optional[drv.CUcontext] = None
        self._green_ctx: Optional[drv.CUgreenCtx] = None
        self._remainder_green_ctx: Optional[drv.CUgreenCtx] = None
        self._stream: Optional[drv.CUstream] = None
        self._remainder_stream: Optional[drv.CUstream] = None
        self._sm_count: Optional[int] = None
        self._remainder_sm_count: Optional[int] = None
        self._total_sms: Optional[int] = None

    def create(self) -> "GreenContext":
        """Create the green context and partition streams.

        Returns self for chaining.

        Raises
        ------
        RuntimeError
            If CUDA driver version < 12.0 or the partition cannot be created.
        ValueError
            If num_sms exceeds device SM count or is below minimum partition size.
        """
        if self._created:
            raise RuntimeError("GreenContext already created")

        # Initialize driver (idempotent)
        _check(drv.cuInit(0))

        # Check driver version
        driver_version = _check(drv.cuDriverGetVersion())
        if driver_version < 12000:
            raise RuntimeError(
                f"CUDA green contexts require driver version >= 12.0, "
                f"got {driver_version // 1000}.{(driver_version % 1000) // 10}"
            )

        # Get device and retain primary context
        self._device = _check(drv.cuDeviceGet(self._device_id))
        self._primary_ctx = _check(drv.cuDevicePrimaryCtxRetain(self._device))
        _check(drv.cuCtxSetCurrent(self._primary_ctx))

        # Query device SM resources
        dev_resource = _check(drv.cuDeviceGetDevResource(
            self._device,
            drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
        ))
        self._total_sms = dev_resource.sm.smCount

        if self._requested_sms >= self._total_sms:
            # Clean up retained context before raising
            drv.cuDevicePrimaryCtxRelease(self._device)
            self._primary_ctx = None
            raise ValueError(
                f"Requested {self._requested_sms} SMs but device only has "
                f"{self._total_sms}. Use fewer SMs to create a partition."
            )

        # Determine alignment
        min_partition = dev_resource.sm.minSmPartitionSize
        alignment = dev_resource.sm.smCoscheduledAlignment

        # Align requested SM count (CUDA 13.0+ requires alignment)
        if alignment > 0 and driver_version >= 13000:
            aligned_sms = (self._requested_sms // alignment) * alignment
            if aligned_sms == 0:
                aligned_sms = alignment
        else:
            aligned_sms = self._requested_sms

        if aligned_sms < min_partition:
            aligned_sms = min_partition

        if aligned_sms != self._requested_sms:
            warnings.warn(
                f"GreenContext: requested {self._requested_sms} SMs, "
                f"aligned to {aligned_sms} (alignment={alignment}, "
                f"min_partition={min_partition})",
                stacklevel=2,
            )

        # Split SMs
        result, _, remainder = _check(drv.cuDevSmResourceSplitByCount(
            1,  # nbGroups: request 1 group
            dev_resource,
            0,  # flags: default behavior
            aligned_sms,
        ))

        self._sm_count = result[0].sm.smCount
        self._remainder_sm_count = remainder.sm.smCount

        # Create resource descriptors
        primary_desc = _check(drv.cuDevResourceGenerateDesc([result[0]], 1))
        remainder_desc = _check(drv.cuDevResourceGenerateDesc([remainder], 1))

        # Create green contexts
        self._green_ctx = _check(drv.cuGreenCtxCreate(
            primary_desc,
            self._device,
            drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
        ))
        self._remainder_green_ctx = _check(drv.cuGreenCtxCreate(
            remainder_desc,
            self._device,
            drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
        ))

        # Create streams within green contexts
        self._stream = _check(drv.cuGreenCtxStreamCreate(
            self._green_ctx,
            drv.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,  # priority: default
        ))
        self._remainder_stream = _check(drv.cuGreenCtxStreamCreate(
            self._remainder_green_ctx,
            drv.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,
        ))

        self._created = True
        return self

    def destroy(self) -> None:
        """Destroy the green context and release all resources. Idempotent."""
        if self._destroyed or not self._created:
            return
        self._destroyed = True

        # Destroy in reverse creation order
        if self._remainder_stream is not None:
            drv.cuStreamDestroy(self._remainder_stream)
            self._remainder_stream = None
        if self._stream is not None:
            drv.cuStreamDestroy(self._stream)
            self._stream = None
        if self._remainder_green_ctx is not None:
            drv.cuGreenCtxDestroy(self._remainder_green_ctx)
            self._remainder_green_ctx = None
        if self._green_ctx is not None:
            drv.cuGreenCtxDestroy(self._green_ctx)
            self._green_ctx = None
        if self._primary_ctx is not None:
            drv.cuDevicePrimaryCtxRelease(self._device)
            self._primary_ctx = None

    def _ensure_created(self) -> None:
        if not self._created:
            raise RuntimeError("GreenContext not yet created — call create() first")
        if self._destroyed:
            raise RuntimeError("GreenContext already destroyed")

    # -- Properties --

    @property
    def stream(self) -> drv.CUstream:
        """Primary partition CUDA stream (CUstream)."""
        self._ensure_created()
        assert self._stream is not None
        return self._stream

    @property
    def remainder_stream(self) -> drv.CUstream:
        """Remainder partition CUDA stream (CUstream)."""
        self._ensure_created()
        assert self._remainder_stream is not None
        return self._remainder_stream

    @property
    def sm_count(self) -> int:
        """Actual SM count of the primary partition (may differ from requested)."""
        self._ensure_created()
        assert self._sm_count is not None
        return self._sm_count

    @property
    def remainder_sm_count(self) -> int:
        """SM count of the remainder partition."""
        self._ensure_created()
        assert self._remainder_sm_count is not None
        return self._remainder_sm_count

    @property
    def total_sms(self) -> int:
        """Total SM count of the device."""
        self._ensure_created()
        assert self._total_sms is not None
        return self._total_sms

    @property
    def torch_stream(self):
        """Primary partition stream wrapped as a PyTorch ExternalStream.

        Use with ``torch.cuda.stream()`` to route FA4 kernels to this partition::

            with torch.cuda.stream(ctx.torch_stream):
                out, lse = flash_attn_func(q, k, v)
        """
        self._ensure_created()
        if torch is None:
            raise ImportError("PyTorch is required for torch_stream property")
        return torch.cuda.ExternalStream(
            int(self._stream),
            device=torch.device("cuda", self._device_id),
        )

    @property
    def remainder_torch_stream(self):
        """Remainder partition stream wrapped as a PyTorch ExternalStream."""
        self._ensure_created()
        if torch is None:
            raise ImportError("PyTorch is required for remainder_torch_stream property")
        return torch.cuda.ExternalStream(
            int(self._remainder_stream),
            device=torch.device("cuda", self._device_id),
        )

    # -- Context manager --

    def __enter__(self) -> "GreenContext":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

    def __del__(self) -> None:
        if self._created and not self._destroyed:
            self.destroy()

    def __repr__(self) -> str:
        if self._created and not self._destroyed:
            return (
                f"GreenContext(primary={self._sm_count} SMs, "
                f"remainder={self._remainder_sm_count} SMs, "
                f"device={self._device_id})"
            )
        state = "destroyed" if self._destroyed else "not created"
        return f"GreenContext(requested={self._requested_sms} SMs, {state})"
