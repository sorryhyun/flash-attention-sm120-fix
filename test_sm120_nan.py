"""Quick diagnostic: compare FA4 SM120 forward/backward vs PyTorch reference."""
import torch

torch.manual_seed(42)
device = "cuda"
dtype = torch.bfloat16

B, S, H, D = 1, 256, 8, 128
q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)

# --- FA4 ---
from flash_attn.cute import flash_attn_func as fa4_func

out_fa4, lse_fa4 = fa4_func(q, k, v)
print(f"FA4 forward  — out has nan: {out_fa4.isnan().any().item()}, inf: {out_fa4.isinf().any().item()}")
print(f"  out stats: min={out_fa4.min().item():.4f}  max={out_fa4.max().item():.4f}  mean={out_fa4.float().mean().item():.4f}")

loss_fa4 = out_fa4.float().sum()
loss_fa4.backward()
print(f"FA4 backward — dq nan: {q.grad.isnan().any().item()}, dk nan: {k.grad.isnan().any().item()}, dv nan: {v.grad.isnan().any().item()}")

# --- PyTorch reference (scaled dot-product) ---
q2 = q.detach().clone().requires_grad_(True)
k2 = k.detach().clone().requires_grad_(True)
v2 = v.detach().clone().requires_grad_(True)

# (B, S, H, D) -> (B, H, S, D)
out_ref = torch.nn.functional.scaled_dot_product_attention(
    q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2)
).transpose(1, 2)

loss_ref = out_ref.float().sum()
loss_ref.backward()

# Compare
max_diff_fwd = (out_fa4.float() - out_ref.float()).abs().max().item()
max_diff_dq = (q.grad.float() - q2.grad.float()).abs().max().item()
max_diff_dk = (k.grad.float() - k2.grad.float()).abs().max().item()
max_diff_dv = (v.grad.float() - v2.grad.float()).abs().max().item()

print(f"\nMax abs diff vs reference:")
print(f"  forward O:  {max_diff_fwd:.6f}")
print(f"  backward dQ: {max_diff_dq:.6f}")
print(f"  backward dK: {max_diff_dk:.6f}")
print(f"  backward dV: {max_diff_dv:.6f}")

# Reasonable tolerance for bf16 attention
ok = max_diff_fwd < 0.05 and max_diff_dq < 0.1 and max_diff_dk < 0.1 and max_diff_dv < 0.1
print(f"\n{'PASS' if ok else 'FAIL'} — {'results look correct' if ok else 'large discrepancy detected'}")
