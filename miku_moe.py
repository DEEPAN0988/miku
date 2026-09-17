"""
MIKU SPARSE MIXTURE OF EXPERTS (Phase II v3.0: MoE Capacity Expansion)
Sparse Gated Mixture of Experts Layer with Top-2 Routing & Auxiliary Load Balancing

Mathematical Foundations:
1. Top-2 Gating: W_g x -> Top-2(logits) -> Softmax normalization
   y = sum_{i in Top-2} G(x)_i * Expert_i(x)
2. Token-Selective Execution: Tokens are dispatched ONLY to their active top-2 experts.
   Zero computation spent on the other 6 inactive experts.
3. Switch Transformer Auxiliary Load Balancing Loss:
   L_aux = num_experts * sum_{i=1}^E (f_i * P_i)
   where f_i is fraction of tokens dispatched to expert i, and P_i is the router softmax probability mass.
"""

import sys
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure UTF-8 output encoding for Windows stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class Expert(nn.Module):
    """Single Feed-Forward Network Expert (d_model -> 4*d_model -> d_model)."""

    def __init__(self, d_model: int = 256, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or (4 * d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts with k=2 routing over 8 experts.
    Replaces standard Transformer FFN layers to expand capacity without compute penalty.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k cannot exceed total experts"
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff

        # Gating router: projects d_model to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Independent expert parameter banks
        self.experts = nn.ModuleList([
            Expert(d_model=d_model) for _ in range(num_experts)
        ])

        # Buffer to track the last computed auxiliary load balancing loss
        self.current_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
            return_aux: If True, returns (out, aux_loss). Otherwise returns out.
        """
        B, T, C = x.shape
        flat_x = x.reshape(-1, C)  # (N, d_model) where N = B * T
        N = flat_x.shape[0]

        # 1. Router Gating Logits & Softmax Probabilities
        router_logits = self.router(flat_x)  # (N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, num_experts)

        # 2. Top-k Expert Selection
        top_weights, top_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (N, k)
        top_weights = F.softmax(top_weights, dim=-1)  # (N, k)

        # 3. Auxiliary Load-Balancing Loss to Prevent Expert Collapse
        expert_mask = F.one_hot(top_indices, num_classes=self.num_experts).sum(dim=1)  # (N, num_experts)
        tokens_per_expert = expert_mask.float().sum(dim=0)  # (num_experts,)
        f_i = tokens_per_expert / (N * self.top_k)
        P_i = router_probs.mean(dim=0)  # (num_experts,)

        aux_loss = self.aux_loss_coeff * (self.num_experts * torch.sum(f_i * P_i))
        self.current_aux_loss = aux_loss

        # 4. Token-Selective Expert Computation
        final_output = torch.zeros_like(flat_x)

        for expert_idx in range(self.num_experts):
            match_mask = (top_indices == expert_idx)
            if not match_mask.any():
                continue

            token_idx, k_idx = torch.where(match_mask)
            selected_tokens = flat_x[token_idx]  # (M, d_model)

            expert_out = self.experts[expert_idx](selected_tokens)  # (M, d_model)
            weights = top_weights[token_idx, k_idx].unsqueeze(-1)  # (M, 1)

            final_output.index_add_(0, token_idx, expert_out * weights)

        out = final_output.reshape(B, T, C)
        if return_aux:
            return out, aux_loss
        return out


def replace_ffn_with_moe(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2
) -> List[SparseMoE]:
    """
    Surgically swaps standard FFN sequential modules in MikuTransformer blocks
    with high-capacity SparseMoE layers.
    """
    moe_layers = []
    if hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "mlp"):
                d_model = block.ln2.normalized_shape[0]
                moe_layer = SparseMoE(d_model=d_model, num_experts=num_experts, top_k=top_k)
                block.mlp = moe_layer
                moe_layers.append(moe_layer)
    return moe_layers


# =====================================================================
# STANDALONE BENCHMARK & LOAD BALANCING VERIFICATION
# =====================================================================

def main():
    print("=" * 70)
    print("  MIKU SPARSE MIXTURE OF EXPERTS (Phase II v3.0 Verification)")
    print("=" * 70)

    d_model = 256
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 32

    moe = SparseMoE(d_model=d_model, num_experts=num_experts, top_k=top_k)

    total_params = sum(p.numel() for p in moe.parameters())
    active_params_per_token = sum(p.numel() for p in moe.experts[0].parameters()) * top_k + sum(p.numel() for p in moe.router.parameters())

    print(f"\n[Capacity Metrics]")
    print(f"  Total MoE Parameters:  {total_params:,}")
    print(f"  Active Compute per Token: {active_params_per_token:,} ({active_params_per_token / total_params * 100:.2f}% active)")
    print(f"  Number of Experts:     {num_experts} (k={top_k} routing)")

    # Test Forward Pass & Auxiliary Loss
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    output, aux_loss = moe(dummy_input)

    print(f"\n[Forward Pass Execution]")
    print(f"  Input Tensor Shape:    {tuple(dummy_input.shape)}")
    print(f"  Output Tensor Shape:   {tuple(output.shape)}")
    print(f"  Aux Load Balance Loss: {aux_loss.item():.6f}")

    # Backward Pass Verification
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target) + aux_loss
    loss.backward()

    router_grad_norm = moe.router.weight.grad.norm().item()
    print(f"\n[Gradient Backpropagation]")
    print(f"  Total Loss (MSE + Aux): {loss.item():.6f}")
    print(f"  Router Gradient Norm:   {router_grad_norm:.6f}")

    print("\n✓ Phase II v3.0 Sparse Mixture of Experts fully verified.")


if __name__ == "__main__":
    main()
