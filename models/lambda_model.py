from typing import Any

import torch
import torch.nn as nn


class LambdaNet(nn.Module):
    """
    Multi-task market price of risk model.

    The model learns a shared date-level representation and exposes:
    - a lambda regression head that predicts the common scalar lambda_t
    - a regime head that predicts whether the future market excess return is
      positive over the target horizon
    - a downside head that predicts negative-regime and tail-downside states
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        max_abs_lambda: float = 0.08,
        dropout: float = 0.10,
        prior_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_abs_lambda = float(max_abs_lambda)
        self.prior = nn.Parameter(torch.tensor([prior_init], dtype=torch.float32))

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.lambda_head = nn.Linear(hidden_dim, 1)
        self.regime_head = nn.Linear(hidden_dim, 1)
        self.downside_head = nn.Linear(hidden_dim, 2)

    def forward(self, state: torch.Tensor) -> dict[str, Any]:
        shared = self.encoder(state)
        lambda_raw = self.lambda_head(shared)
        lambda_t = self.prior + self.max_abs_lambda * torch.tanh(lambda_raw)
        sign_logit = self.regime_head(shared)
        downside_logits = self.downside_head(shared)
        return {
            "lambda_t": lambda_t.squeeze(-1),
            "sign_logit": sign_logit.squeeze(-1),
            "negative_logit": downside_logits[:, 0],
            "tail_logit": downside_logits[:, 1],
            "shared": shared,
        }
