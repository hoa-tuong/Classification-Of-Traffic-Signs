import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss danh cho bai toan phan loai da lop (Multi-class Classification)
    giup tap trung vao cac mau kho (hard examples) va giam trong so cua cac mau de.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss ket hop Label Smoothing giup chong overfitting
    va giam do tu tin thai qua tren cac bien bao co hinh dang tuong dong.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, label_smoothing=self.smoothing)


def get_loss_fn(loss_type: str = "cross_entropy", **kwargs) -> nn.Module:
    """
    Factory function khoi tao ham mat mat phu hop cho do an.

    Args:
        loss_type: 'cross_entropy', 'focal_loss', hoac 'label_smoothing'.
    """
    loss_lower = loss_type.lower()
    if loss_lower in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    elif loss_lower in ("focal", "focal_loss"):
        gamma = kwargs.get("gamma", 2.0)
        alpha = kwargs.get("alpha", None)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_lower in ("smoothing", "label_smoothing"):
        smoothing = kwargs.get("smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        raise ValueError(f"Khong ho tro loss_type='{loss_type}'. Chon 'cross_entropy', 'focal_loss', hoac 'label_smoothing'.")
