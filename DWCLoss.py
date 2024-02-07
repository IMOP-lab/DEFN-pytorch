from __future__ import annotations
import warnings
from collections.abc import Callable, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction,look_up_option, pytorch_after

def softmax_FocalLoss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None  
) -> torch.Tensor:
    input_ls = input.log_softmax(1)
    loss: torch.Tensor = -(1 - input_ls.exp()).pow(gamma) * input_ls * target
    if alpha is not None:
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss
    return loss

class DeepRankingLoss(nn.Module):
  def __init__(self, margin=0.1):
    super(DeepRankingLoss, self).__init__()
    self.margin = margin
  def forward(self, positive, negative):
    distance_positive = (positive - positive.mean()).pow(2).sum()
    distance_negative = (negative - negative.mean()).pow(2).sum()
    loss = torch.clamp(self.margin + distance_positive - distance_negative, min=0.0)
    return loss
  
class FocalLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        use_softmax: bool = True,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.use_softmax = use_softmax
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_pred_ch = input.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("ignored")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("ignored")
            else:
                target = target[:, 1:]
                input = input[:, 1:]
        if target.shape != input.shape:
            raise ValueError("ground truth has different shape")
        loss: Optional[torch.Tensor] = None
        input = input.float()
        target = target.float()
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("ignored")
            loss = softmax_FocalLoss(input, target, self.gamma, self.alpha)
        if self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        else:
            raise ValueError("Unsupported reduction")
        return loss

class DiceLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("ignored")
            else:
                input = torch.softmax(input, 1)
        if self.other_act is not None:
            input = self.other_act(input)
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("ignored")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("ignored")
            else:
                target = target[:, 1:]
                input = input[:, 1:]
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, dim=reduce_axis)
        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)
        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  
        else:
            raise ValueError("Unsupported reduction")
        return f

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, inputs, targets):
        targets = one_hot(targets, self.num_classes)
        inputs_boundary = F.avg_pool3d(inputs, kernel_size=3, stride=1, padding=1) - inputs
        targets_boundary = F.avg_pool3d(targets, kernel_size=3, stride=1, padding=1) - targets
        boundary_loss = F.mse_loss(inputs_boundary, targets_boundary)
        return boundary_loss

class DWCLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_boundary: float = 1.0,
        lambda_focal: float = 1.0, 
        weight = [1, 2, 3, 4],
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.focal = FocalLoss()
        self.deep_ranking_loss = DeepRankingLoss()
        self.boundary = BoundaryLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        if lambda_focal < 0.0: 
            raise ValueError("lambda_focal should be no less than 0.0.")
        if lambda_boundary < 0.0: 
            raise ValueError("lambda_boundary should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_boundary = lambda_boundary
        self.lambda_focal = lambda_focal
        self.old_pt_ver = not pytorch_after(1, 10)
    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn("version error")
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)
        return self.cross_entropy(input, target)  

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same.")
        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        boundary_loss = self.boundary(input, target)*100
        focal_loss = self.focal(input, target)*10
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss + self.lambda_boundary * boundary_loss + self.lambda_focal * focal_loss
        if total_loss <= 0.4:
            self.lambda_dice = 0.2
            self.lambda_ce = 0.2
            self.lambda_boundary = 0.3
            self.lambda_focal = 0.3
        elif total_loss <= 0.5:
            self.lambda_dice = 0.3
            self.lambda_ce = 0.25
            self.lambda_boundary = 0.2
            self.lambda_focal = 0.25
        elif total_loss <= 0.6:
            self.lambda_dice = 0.4
            self.lambda_ce = 0.25
            self.lambda_boundary = 0.1
            self.lambda_focal = 0.25
        elif total_loss <= 0.8:
            self.lambda_dice = 0.45
            self.lambda_ce = 0.25
            self.lambda_boundary = 0.1
            self.lambda_focal = 0.2
        # if total_loss <= 0.5:
        #     self.lambda_dice = 0.3
        #     self.lambda_ce = 0.2
        #     self.lambda_boundary = 0.2
        #     self.lambda_focal = 0.3
        # elif total_loss <= 0.6:
        #     self.lambda_dice = 0.35
        #     self.lambda_ce = 0.25
        #     self.lambda_boundary = 0.15
        #     self.lambda_focal = 0.25
        # elif total_loss <= 0.8:
        #     self.lambda_dice = 0.4
        #     self.lambda_ce = 0.25
        #     self.lambda_boundary = 0.1
        #     self.lambda_focal = 0.25
        # elif total_loss <= 1:
        #     self.lambda_dice = 0.45
        #     self.lambda_ce = 0.25
        #     self.lambda_boundary = 0.1
        #     self.lambda_focal = 0.2
        positive = total_loss[total_loss < total_loss.mean()]
        negative = total_loss[total_loss >= total_loss.mean()]
        ranking_loss = self.deep_ranking_loss(positive, negative)
        total_loss += ranking_loss
        return total_loss
