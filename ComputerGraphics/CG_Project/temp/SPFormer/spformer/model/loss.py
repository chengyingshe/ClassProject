import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from scipy.optimize import linear_sum_assignment
from typing import Optional


@torch.jit.script
def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob)**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


@torch.jit.script
def dice_loss_multi_calsses(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float] = None) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from
    https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    input = input.permute(1, 0)
    target = target.permute(1, 0)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1.0 - per_channel_dice

    return loss.mean()


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))

    @torch.no_grad()
    def forward(self, pred_labels, pred_masks, insts):
        '''
        pred_masks: List[Tensor] len(p2c) == B, Tensor.shape == (n, N)
        pred_labels: (B, n_q, 19)
        insts: List[Instances3D]
        '''
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)  # (n_q, 19)
            tgt_idx = inst.gt_labels  # (num_inst,)
            cost_class = -pred_label[:, tgt_idx]  # (n_q, num_inst)

            tgt_mask = inst.gt_spmasks  # (num_inst, N)

            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        ignore_label=-100,
        loss_weight=[1.0, 1.0, 1.0, 1.0],
        cost_weight=[1.0, 1.0, 1.0],
        non_object_weight=0.1,
        num_class=18,
    ):
        super().__init__()
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer('class_weight', class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_class = num_class
        self.ignore_label = ignore_label

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_inst_info(self, batched_gt_instance, coords, batch_offsets):
        for i, gt_inst in enumerate(batched_gt_instance):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            coord = coords[start_id:end_id]  # (N, 3)
            inst_idx, point_idx = torch.nonzero(gt_inst['gt_masks'], as_tuple=True)
            inst_point = coord[point_idx]
            gt_inst['gt_center'] = torch_scatter.segment_coo(inst_point, inst_idx.cuda(), reduce='mean')

    def get_layer_loss(self, layer, aux_outputs, insts):
        loss_out = {}
        pred_labels = aux_outputs['labels']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out['cls_loss'] = class_loss.item()

        # # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)
        mask_dice_loss = mask_dice_loss / len(pred_masks)

        loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        loss = (
            self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss)

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def forward(self, pred, insts):
        '''
        pred_masks: List[Tensor (n, M)]
        pred_labels: (B, n, 19)
        pred_scores: (B, n, 1) or [(B, n, 1)]
        insts: List[Instance3D]
        '''
        loss_out = {}

        pred_labels = pred['labels']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out['cls_loss'] = class_loss.item()

        # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)

        loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        loss = (
            self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, insts)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss.item()

        return loss, loss_out


@gorilla.LOSSES.register_module()
class EnhancedCriterion(nn.Module):
    def __init__(self, 
                ignore_label=-100,
                loss_weight=[1.0, 1.0, 1.0, 1.0],
                cost_weight=[1.0, 1.0, 1.0],
                non_object_weight=0.1,
                num_class=18):
        super().__init__()
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer('class_weight', class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_class = num_class
        self.ignore_label = ignore_label
        
        # Additional boundary loss weight
        self.boundary_weight = 0.5
        self.feature_weight = 0.3
        
    def compute_boundary_loss(self, pred_masks, gt_masks):
        """Compute boundary-aware loss"""
        # Get mask boundaries using gradient
        pred_grad = self.compute_gradient(pred_masks)
        gt_grad = self.compute_gradient(gt_masks)
        
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred_grad, gt_grad.float(), reduction='mean'
        )
        return boundary_loss
        
    def compute_gradient(self, masks):
        """Compute mask gradients for boundary detection"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              device=masks.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              device=masks.device).float()
        
        grad_x = F.conv2d(masks.unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0))
        grad_y = F.conv2d(masks.unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0))
        
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2))

    def compute_feature_loss(self, pred_features, instance_labels):
        """Compute feature embedding loss"""
        feature_loss = 0
        unique_instances = torch.unique(instance_labels)
        
        for inst_id in unique_instances:
            if inst_id == self.ignore_label:
                continue
                
            # Get features for current instance
            inst_mask = (instance_labels == inst_id)
            inst_features = pred_features[inst_mask]
            
            if inst_features.shape[0] == 0:
                continue
                
            # Compute instance center
            center = torch.mean(inst_features, dim=0)
            
            # Pull loss: pull features to instance center
            pull_loss = torch.mean((inst_features - center).norm(p=2, dim=1))
            
            # Push loss: push different instance centers apart
            push_loss = 0
            for other_id in unique_instances:
                if other_id == inst_id or other_id == self.ignore_label:
                    continue
                    
                other_mask = (instance_labels == other_id)
                other_center = torch.mean(pred_features[other_mask], dim=0)
                
                dist = torch.norm(center - other_center, p=2)
                push_loss += torch.exp(-dist)
            
            feature_loss += (pull_loss + push_loss)
            
        return feature_loss / (len(unique_instances) + 1e-6)

    def forward(self, pred, insts):
        loss_out = {}
        
        # Original classification and mask losses
        class_loss, mask_bce_loss, mask_dice_loss, score_loss = self.compute_base_losses(
            pred, insts
        )
        
        # Boundary loss
        boundary_loss = 0
        for mask, inst in zip(pred['masks'], insts):
            boundary_loss += self.compute_boundary_loss(mask, inst.gt_spmasks)
        boundary_loss = boundary_loss / len(pred['masks'])
        
        # Feature embedding loss
        feature_loss = 0
        if 'features' in pred:
            for features, inst in zip(pred['features'], insts):
                feature_loss += self.compute_feature_loss(
                    features, inst.gt_instance_labels
                )
            feature_loss = feature_loss / len(pred['features'])
        
        # Combine all losses
        loss = (self.loss_weight[0] * class_loss + 
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss + 
                self.loss_weight[3] * score_loss +
                self.boundary_weight * boundary_loss +
                self.feature_weight * feature_loss)
        
        # Update loss dict
        loss_out.update({
            'cls_loss': class_loss.item(),
            'mask_bce_loss': mask_bce_loss.item(),
            'mask_dice_loss': mask_dice_loss.item(),
            'score_loss': score_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'feature_loss': feature_loss.item(),
            'loss': loss.item()
        })
        
        return loss, loss_out