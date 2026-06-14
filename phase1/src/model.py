"""
phase1/src/model.py
EfficientNet-B2 with an ordinal classification head.

Loss
----
CORN loss (Conditional Ordinal Regression with Neural Networks) decomposes
an ordinal K-class problem into K-1 binary sub-problems:

    Task k: P(Y > k | Y >= k)   for k = 0, 1, …, K-2

Each sub-task is trained with weighted binary cross-entropy.
The loss is averaged across all K-1 tasks.

Reference: Shi et al., "Deep Neural Networks for Rank-Consistent Ordinal
Regression Based on Conditional Probabilities", arXiv 2111.08851.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Model ───────────────────────────────────────────────────────────────────

class EfficientNetB2Ordinal(nn.Module):
    """
    EfficientNet-B2 backbone with a dropout + linear classification head.

    The head outputs ``num_classes`` raw logits.  These are interpreted as
    CORN conditional logits, not standard softmax logits, so:
        - During training: pass to ``corn_loss()``.
        - During inference: pass to ``corn_label_to_probs()`` to get
          proper class probabilities, then argmax.

    Parameters
    ----------
    num_classes : int
        Number of ordinal target classes (default 3 for mild/moderate/severe).
    dropout : float
        Dropout probability before the linear head.
    pretrained : bool
        Load ImageNet weights via timm.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm  # imported lazily so tests without timm still import the module

        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,          # strip original head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1408 for B2

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, C, H, W]

        Returns
        -------
        logits : Tensor [B, num_classes]
            Raw (un-activated) CORN logits.
        """
        features = self.backbone(x)     # [B, in_features]
        return self.head(features)      # [B, num_classes]


# ─── CORN Loss ───────────────────────────────────────────────────────────────

def corn_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    weights: torch.FloatTensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    CORN (Conditional Ordinal Regression with Neural Networks) loss.

    For each rank boundary k ∈ {0, …, K-2} this computes a binary
    cross-entropy for the conditional probability P(Y > k | Y >= k).

    Only samples with Y >= k participate in task k (conditioning).

    Parameters
    ----------
    logits : Tensor [B, K]
        Raw logits from the model (NOT sigmoid-ed).
    targets : Tensor [B]
        Integer class labels in [0, K-1].
    num_classes : int
        K — total number of ordinal classes.
    weights : FloatTensor [K] | None
        Per-class weights.  Scaled to per-task weights automatically.
    label_smoothing : float
        Applied to the binary BCE targets (slight smoothing).

    Returns
    -------
    loss : scalar Tensor
    """
    sets = []
    n = num_classes - 1  # number of binary tasks

    for i in range(n):
        # Subset: only samples eligible for rank-i task (label >= i)
        label_mask = targets >= i
        logits_task = logits[label_mask, i]          # [S]
        binary_labels = (targets[label_mask] > i).float()  # 1 if Y > i, else 0

        if logits_task.numel() == 0:
            continue  # degenerate batch

        # Label smoothing
        if label_smoothing > 0.0:
            binary_labels = binary_labels * (1 - label_smoothing) + 0.5 * label_smoothing

        # Per-sample weighting derived from class weights
        if weights is not None:
            # Assign weight based on the original class label
            sample_weights = weights.to(logits.device)[targets[label_mask]]
        else:
            sample_weights = None

        loss_i = F.binary_cross_entropy_with_logits(
            logits_task,
            binary_labels,
            weight=sample_weights,
            reduction="mean",
        )
        sets.append(loss_i)

    if not sets:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    return torch.stack(sets).mean()


# ─── Inference helpers ────────────────────────────────────────────────────────

def corn_label_to_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert CORN logits to class probabilities.

    Uses the cumulative-product decomposition:
        P(Y=0) = 1 - σ(l_0)
        P(Y=k) = σ(l_{k-1}) * ... * σ(l_0) * (1 - σ(l_k))   for 0 < k < K-1
        P(Y=K-1) = σ(l_{K-2}) * ... * σ(l_0)

    Parameters
    ----------
    logits : Tensor [B, K]
    num_classes : int

    Returns
    -------
    probs : Tensor [B, K]  (rows sum to 1)
    """
    sig = torch.sigmoid(logits)          # [B, K]
    probs = torch.zeros_like(sig)

    # Cumulative product of "exceeds rank k" probabilities
    # cum_prod[i] = prod_{j < i} sig[:, j]
    cum_prod = torch.ones(sig.shape[0], device=sig.device)

    for k in range(num_classes):
        if k == 0:
            probs[:, k] = 1.0 - sig[:, 0]
        elif k == num_classes - 1:
            probs[:, k] = cum_prod * sig[:, k - 1]
        else:
            probs[:, k] = cum_prod * (1.0 - sig[:, k])

        if k < num_classes - 1:
            cum_prod = cum_prod * sig[:, k]

    # Clamp for numerical stability and re-normalise
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def predict(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return predicted class indices from raw CORN logits."""
    probs = corn_label_to_probs(logits, num_classes)
    return probs.argmax(dim=1)


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_model(config: dict) -> EfficientNetB2Ordinal:
    """Instantiate model from phase1.yaml config dict."""
    mcfg = config["model"]
    return EfficientNetB2Ordinal(
        num_classes=mcfg["num_classes"],
        dropout=mcfg.get("dropout", 0.3),
        pretrained=mcfg.get("pretrained", True),
    )
