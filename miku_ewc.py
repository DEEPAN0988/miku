"""
MIKU ELASTIC WEIGHT CONSOLIDATION (Phase II v4.0: Continual Learning)
Prevents Catastrophic Forgetting via Diagonal Fisher Information Matrix Regularization

Mathematical Formulation:
1. Empirical Fisher Information Matrix Diagonals:
   F_i = (1 / N) * sum_{n=1}^N ( d/d(theta_i) log p(y_n | x_n) )^2
2. EWC Elastic Penalty:
   L_EWC = (lambda / 2) * sum_i F_i * (theta_i - theta_{i,old})^2
3. Hardware Sympathy: Computed EXCLUSIVELY over active LoRA parameters (65,536 weights).
   Base 8M canonical parameters remain frozen with zero memory overhead.
"""

import sys
from typing import Dict, List, Tuple, Optional
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


class ElasticWeightConsolidation:
    """
    Manages Fisher Information Matrix diagonals and anchor weights
    to penalize deviation from previously consolidated cognitive skills.
    """

    def __init__(self, lambda_ewc: float = 0.4):
        self.lambda_ewc = lambda_ewc
        self.fisher_diagonals: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}
        self.is_consolidated: bool = False

    def consolidate(
        self,
        model: nn.Module,
        calibration_data: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Computes empirical Fisher Information diagonals over active parameters:
        F_i = E_x [ (grad_theta_i log p(x))^2 ]
        """
        model.eval()
        # Collect only parameters that require gradients (e.g. LoRA parameters)
        trainable_params = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        if not trainable_params:
            return

        # Initialize Fisher accumulator
        fisher_accum: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, device=p.device)
            for name, p in trainable_params.items()
        }

        # Cache optimal anchor weights theta_{old}
        self.optimal_weights = {
            name: p.detach().clone()
            for name, p in trainable_params.items()
        }

        N = len(calibration_data)
        if N == 0:
            return

        for input_ids, target_ids in calibration_data:
            model.zero_grad()
            logits = model(input_ids)  # (B, T, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Gather target token log-likelihood
            target_log_probs = torch.gather(
                log_probs, dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            loss = -target_log_probs.mean()
            loss.backward()

            with torch.no_grad():
                for name, p in trainable_params.items():
                    if p.grad is not None:
                        fisher_accum[name] += (p.grad.detach() ** 2) / N

        self.fisher_diagonals = fisher_accum
        self.is_consolidated = True

    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Calculates EWC quadratic loss penalty:
        L_EWC = (lambda / 2) * sum_i F_i * (theta_i - theta_{i,old})^2
        """
        if not self.is_consolidated or not self.fisher_diagonals:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, p in model.named_parameters():
            if name in self.fisher_diagonals:
                fisher = self.fisher_diagonals[name]
                theta_old = self.optimal_weights[name]
                diff = p - theta_old
                ewc_loss = ewc_loss + torch.sum(fisher * (diff ** 2))

        return 0.5 * self.lambda_ewc * ewc_loss


# =====================================================================
# STANDALONE EWC VERIFICATION
# =====================================================================

def main():
    print("=" * 70)
    print("  MIKU ELASTIC WEIGHT CONSOLIDATION (Phase II v4.0 Verification)")
    print("=" * 70)

    # 1. Initialize toy parameter network
    linear = nn.Linear(32, 16)
    ewc = ElasticWeightConsolidation(lambda_ewc=0.5)

    # 2. Synthetic task A (Python code synthesis task)
    calib_inputs = [
        (torch.randint(0, 32, (1, 8)), torch.randint(0, 16, (1, 8)))
        for _ in range(5)
    ]

    print("\n[Consolidation] Computing Fisher Information Diagonals for Task A...")
    # Wrap linear with dummy forward
    class DummyModel(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
        def forward(self, idx):
            # One-hot embedding approximation
            x = F.one_hot(idx, num_classes=32).float()
            return self.layer(x)

    dummy_model = DummyModel(linear)
    ewc.consolidate(dummy_model, calib_inputs)

    print(f"  Fisher Diagonals Computed: {list(ewc.fisher_diagonals.keys())}")
    for name, f in ewc.fisher_diagonals.items():
        print(f"  {name}: Mean Fisher Sensitivity = {f.mean().item():.6f}")

    # 3. Verify zero loss at optimal weights
    initial_ewc_loss = ewc.compute_ewc_loss(dummy_model)
    print(f"\n[EWC Loss at Anchor Point]: {initial_ewc_loss.item():.6f} (Expected: 0.0)")
    assert initial_ewc_loss.item() == 0.0

    # 4. Perturb weights to simulate learning Task B (UI Navigation)
    with torch.no_grad():
        for p in dummy_model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    perturbed_ewc_loss = ewc.compute_ewc_loss(dummy_model)
    print(f"[EWC Penalty on Perturbed Weights]: {perturbed_ewc_loss.item():.6f} (Penalty Active)")
    assert perturbed_ewc_loss.item() > 0.0

    print("\n✓ Phase II v4.0 Elastic Weight Consolidation fully verified.")


if __name__ == "__main__":
    main()
