"""
Template metrics for CH segmentation.

CHThresholdMetrics defines four metric sets:
- "train_loss"    — differentiable loss that drives backpropagation (BCE + soft Dice,
                    chosen for the CH-pixel class imbalance).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints.
                    Delegates to "train_loss" so the monitored objective can't drift from
                    the training objective by accident.
- "train_metrics" — non-differentiable metrics logged during training (IoU, Dice
                    coefficient) computed on the hard (thresholded) mask.
- "val_metrics"   — same hard IoU/Dice, reported at validation only. These do NOT
                    influence checkpoint selection — "val_loss" does.

The __call__ method selects the appropriate metric set based on the mode passed at
construction time. The dictionary keys returned by each method become the metric names
propagated to the logger (e.g. WandB, CSV).
"""

import torch
import torch.nn.functional as F


# Shape contract: preds come from ThresholdCHModel.forward as logits of shape (B, H, W),
# target is the CH mask of the same shape. Both are flattened with
# reshape(preds.shape[0], -1) rather than squeeze(), for the same batch-of-one reason as
# elsewhere in this repo: squeeze is shape-dependent and would collapse a batch of one.
class CHThresholdMetrics:
    def __init__(self, mode: str, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        """
        Initialize CHThresholdMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. One of "train_loss",
                        "val_loss", "train_metrics", or "val_metrics".
            bce_weight (float): Weight applied to the BCE term in "train_loss"/"val_loss".
            dice_weight (float): Weight applied to the soft Dice term in "train_loss"/"val_loss".
            eps (float): Numerical stability guard for the Dice ratios (also covers the
                        empty-mask edge case).
        """
        self.mode = mode
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def _soft_dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Differentiable Dice loss computed on sigmoid probabilities."""
        probs = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        intersection = (probs * target).sum()
        dice = (2 * intersection + self.eps) / (probs.sum() + target.sum() + self.eps)

        return 1 - dice

    def _hard_iou_dice(self, preds: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-differentiable IoU and Dice coefficient on the thresholded (hard) mask."""
        probs = torch.sigmoid(preds).reshape(preds.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        pred_mask = (probs > 0.5).float()

        intersection = (pred_mask * target).sum()
        union = pred_mask.sum() + target.sum() - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        dice_coef = (2 * intersection + self.eps) / (pred_mask.sum() + target.sum() + self.eps)

        return iou, dice_coef

    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training: BCE (on logits) + soft Dice.

        Args:
            preds (torch.Tensor): Model logits, shape (B, H, W).
            target (torch.Tensor): Ground truth CH mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                                        Keys are metric names ("bce", "dice"), and values are
                                        the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        target = target.float()

        output_metrics = {}
        output_weights = []

        output_metrics["bce"] = F.binary_cross_entropy_with_logits(preds, target)
        output_weights.append(self.bce_weight)

        output_metrics["dice"] = self._soft_dice_loss(preds, target)
        output_weights.append(self.dice_weight)

        return output_metrics, output_weights

    def val_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate the validation loss — the quantity logged as ``val_loss`` and used by
        ModelCheckpoint to select the best model.

        By default this delegates to ``train_loss``, so the monitored quantity has the same
        form as the training objective and the two cannot drift apart by accident. This is
        the hook to override when your task needs a different validation objective (a
        different weighting, a metric that is meaningful only on held-out data, etc.).

        Note that this is deliberately separate from ``val_metrics``: those are reported
        for information only and do not affect checkpoint selection.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                - list[float]: List of weights for each calculated metric.
        """
        return self.train_loss(preds, target)

    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training: hard-mask IoU and Dice coefficient.
        IMPORTANT:  These metrics are only for reporting purposes and do not
                    contribute to the training loss. Use only if you want to
                    monitor additional metrics during training.

        Args:
            preds (torch.Tensor): Model logits, shape (B, H, W).
            target (torch.Tensor): Ground truth CH mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated evaluation metrics.
                                        Keys are metric names ("iou", "dice_coef"), and values are
                                        the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        iou, dice_coef = self._hard_iou_dice(preds, target)

        return {"iou": iou, "dice_coef": dice_coef}, [1, 1]

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation: hard-mask IoU and Dice coefficient.

        Args:
            preds (torch.Tensor): Model logits, shape (B, H, W).
            target (torch.Tensor): Ground truth CH mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated metrics.
                                        Keys are metric names ("iou", "dice_coef"), and values are
                                        the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        iou, dice_coef = self._hard_iou_dice(preds, target)

        return {"iou": iou, "dice_coef": dice_coef}, [1, 1]

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Evaluate metrics for the mode set at construction time.

        Args:
            preds: Model output tensor. Shape depends on the application.
            target: Ground truth tensor to compare against.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Metric dictionary. Keys become logger metric names; values are
                  scalar tensors aggregated over the batch.
                - List of per-metric weights (used by FlareLightningModule to
                  combine multiple loss terms into a single scalar).
        """

        match self.mode.lower():

            case "train_loss":
                return self.train_loss(preds, target)

            # No torch.no_grad() here, matching "train_loss": Lightning already disables
            # gradients during validation, so wrapping it would differ gratuitously from
            # the loss case this mirrors.
            case "val_loss":
                return self.val_loss(preds, target)

            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)

            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)

            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )
