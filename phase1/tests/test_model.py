"""
phase1/tests/test_model.py
Tests for EfficientNetB2Ordinal, corn_loss, and corn_label_to_probs.

These tests use pretrained=False to avoid downloading weights.
They require torch but NOT a GPU (all run on CPU).
"""
from __future__ import annotations

import pytest
import torch


class TestCornLoss:
    """Test the CORN ordinal loss function."""

    def test_zero_for_perfect_predictions(self) -> None:
        """Loss should be very low when logits perfectly rank the classes."""
        from phase1.src.model import corn_loss
        # logits ordered so sigmoid > 0.5 for the right cut-points
        # label=0 → only sigmoid(l0) should be small
        # label=2 → both cut-points should be exceeded
        logits = torch.tensor([
            [-5.0, -5.0, -5.0],   # pred = 0 (mild)
            [5.0,  -5.0, -5.0],   # pred = 1 (moderate)
            [5.0,   5.0,  5.0],   # pred = 2 (severe)
        ])
        targets = torch.tensor([0, 1, 2])
        loss = corn_loss(logits, targets, num_classes=3)
        assert loss.item() < 0.5

    def test_high_for_wrong_predictions(self) -> None:
        from phase1.src.model import corn_loss
        # Completely reversed logits
        logits = torch.tensor([
            [5.0,  5.0,  5.0],    # model says severe for mild
            [-5.0, -5.0, -5.0],   # model says mild for moderate
            [-5.0, -5.0, -5.0],   # model says mild for severe
        ])
        targets = torch.tensor([0, 1, 2])
        loss = corn_loss(logits, targets, num_classes=3)
        assert loss.item() > 1.0

    def test_gradient_flows(self) -> None:
        from phase1.src.model import corn_loss
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))
        loss = corn_loss(logits, targets, num_classes=3)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_weighted_loss_differs(self) -> None:
        from phase1.src.model import corn_loss
        logits = torch.randn(12, 3)
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        loss_unweighted = corn_loss(logits, targets, num_classes=3, weights=None)
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_weighted = corn_loss(logits, targets, num_classes=3, weights=weights)
        # They should differ (not guaranteed to be larger/smaller — just different)
        assert abs(loss_unweighted.item() - loss_weighted.item()) > 1e-6

    def test_empty_batch_does_not_crash(self) -> None:
        from phase1.src.model import corn_loss
        # Single sample degenerate batch
        logits = torch.randn(1, 3)
        targets = torch.tensor([0])
        loss = corn_loss(logits, targets, num_classes=3)
        assert torch.isfinite(loss)


class TestCornLabelToProbs:
    """Test probability conversion from CORN logits."""

    def test_probs_sum_to_one(self) -> None:
        from phase1.src.model import corn_label_to_probs
        logits = torch.randn(16, 3)
        probs = corn_label_to_probs(logits, num_classes=3)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)

    def test_probs_non_negative(self) -> None:
        from phase1.src.model import corn_label_to_probs
        logits = torch.randn(16, 3)
        probs = corn_label_to_probs(logits, num_classes=3)
        assert (probs >= 0).all()

    def test_high_logit_predicts_highest_class(self) -> None:
        from phase1.src.model import corn_label_to_probs, predict
        # Very high logits → model should predict class 2
        logits = torch.full((4, 3), 10.0)
        preds = predict(logits, num_classes=3)
        assert (preds == 2).all()

    def test_very_negative_logit_predicts_lowest_class(self) -> None:
        from phase1.src.model import corn_label_to_probs, predict
        logits = torch.full((4, 3), -10.0)
        preds = predict(logits, num_classes=3)
        assert (preds == 0).all()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("timm"),
    reason="timm not installed",
)
class TestEfficientNetB2Ordinal:
    """Integration tests for the full model (requires timm)."""

    def test_forward_shape(self) -> None:
        from phase1.src.model import EfficientNetB2Ordinal
        model = EfficientNetB2Ordinal(num_classes=3, dropout=0.0, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 260, 260)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3)

    def test_get_model_factory(self) -> None:
        from phase1.src.model import get_model
        config = {
            "model": {
                "backbone": "efficientnet_b2",
                "pretrained": False,
                "num_classes": 3,
                "dropout": 0.1,
            }
        }
        model = get_model(config)
        assert model.num_classes == 3
