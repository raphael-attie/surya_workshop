"""
Template metrics for coronal hole segmentation.

FlareMetrics defines four metric sets:
- "train_loss"    — differentiable loss that drives backpropagation (MSE).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints.
                    Defaults to the same MSE as "train_loss"; override it when your task
                    needs a different validation objective.
- "train_metrics" — non-differentiable metrics logged during training (RRSE).
- "val_metrics"   — metrics logged at validation for reporting only (MSE + RRSE). These
                    do NOT influence checkpoint selection — "val_loss" does.

The __call__ method selects the appropriate metric set based on the mode passed at
construction time. The dictionary keys returned by each method become the metric names
propagated to the logger (e.g. WandB, CSV).
"""

import torch
import torch.nn.functional as F
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

# Shape contract: predictions arrive as (B,) from HelioSpectformer1D or (B, 1) from the
# linear baseline, while targets are always (B, 1). Every metric below flattens both with
# reshape(-1) rather than squeeze(-1): squeeze is shape-dependent and collapses a
# batch of one to a 0-d scalar, which then fails to broadcast against a (1,) target.
class CHThresholdMetrics:
    def __init__(self, mode: str, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        """
        Initialize CHThresholdMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. One of "train_loss",
                        "val_loss", "train_metrics", or "val_metrics".
            bce_weight (float): Weight applied to the BCE term in train_loss/val_loss.
            dice_weight (float): Weight applied to the soft-Dice term in train_loss/val_loss.
            eps (float): Numerical-stability constant added to both the numerator and
                        denominator of every Dice/IoU ratio, so a sample with no predicted
                        and no true CH pixels (0/0) scores a perfect match instead of NaN.
        """
        self.mode = mode
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def _soft_dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft-Dice loss computed from raw logits.

        Args:
            logits (torch.Tensor): Model output logits, shape (B, H, W).
            target (torch.Tensor): Ground truth mask, shape (B, H, W).

        Returns:
            torch.Tensor: Scalar ``1 - dice``, averaged over the batch.
        """
        probs = torch.sigmoid(logits)

        probs_flat = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.eps) / (
            probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        )

        return (1 - dice).mean()


    def train_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training: binary cross-entropy + soft-Dice.

        Args:
            preds (torch.Tensor): Model output logits, shape (B, H, W).
            target (torch.Tensor): Ground truth mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: {"bce": ..., "dice": ...}.
                - list[float]: [self.bce_weight, self.dice_weight].
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

        Delegates to ``train_loss`` (same BCE + soft-Dice objective), matching
        ``FlareMetrics.val_loss``'s default-delegation pattern (from the template).

        Args:
            preds (torch.Tensor): Model output logits, shape (B, H, W).
            target (torch.Tensor): Ground truth mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]: Same shape as ``train_loss``.
        """
        return self.train_loss(preds, target)

    def _hard_iou_dice(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Non-differentiable IoU and Dice coefficient computed from a hard-thresholded mask.

        Args:
            preds (torch.Tensor): Model output logits, shape (B, H, W).
            target (torch.Tensor): Ground truth mask (already float), shape (B, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (iou, dice_coef), each a scalar averaged
            over the batch.
        """
        probs = torch.sigmoid(preds)
        pred_mask = (probs > 0.5).float()

        pred_flat = pred_mask.reshape(pred_mask.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        union = pred_sum + target_sum - intersection

        # eps guards the empty/empty case (no predicted or true CH pixels in a sample):
        # union == 0 and intersection == 0 would otherwise divide 0/0 -> NaN. With eps,
        # both iou and dice_coef evaluate to 1.0 for that sample instead.
        iou = (intersection + self.eps) / (union + self.eps)
        dice_coef = (2 * intersection + self.eps) / (pred_sum + target_sum + self.eps)

        return iou.mean(), dice_coef.mean()
    
    def train_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training: hard IoU and Dice coefficient.
        IMPORTANT:  These metrics are only for reporting purposes and do not
                    contribute to the training loss.

        Args:
            preds (torch.Tensor): Model output logits, shape (B, H, W).
            target (torch.Tensor): Ground truth mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: {"iou": ..., "dice_coef": ...}.
                - list[float]: [1, 1].
        """
        iou, dice_coef = self._hard_iou_dice(preds, target.float())
        return {"iou": iou, "dice_coef": dice_coef}, [1, 1]

    def val_metrics(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation: hard IoU and Dice coefficient.

        Delegates to ``train_metrics`` — the spec has train_metrics and val_metrics report
        the identical {"iou", "dice_coef"} pair here (unlike FlareMetrics.val_metrics,
        which adds an extra term), so there's nothing to compute independently.

        Args:
            preds (torch.Tensor): Model output logits, shape (B, H, W).
            target (torch.Tensor): Ground truth mask, shape (B, H, W).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]: Same shape as ``train_metrics``.
        """

        return self.train_metrics(preds, target)

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Evaluate metrics for the mode set at construction time.

        Args:
            preds: Model output logits, shape (B, H, W).
            target: Ground truth mask tensor to compare against, shape (B, H, W).

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
