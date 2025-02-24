import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple

class LovaszHingeLoss(nn.Module):
    """Binary Lovasz hinge loss."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute gradient of the Lovasz extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, logits, labels):
        """
        Binary Lovasz hinge loss.
        Args:
            logits: [N] Logits at each prediction (between -infty and +infty)
            labels: [N] Binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            return torch.tensor(0.).cuda()
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

class GIoULoss(nn.Module):
    """Generalized IoU loss for masks."""
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        pred_masks = pred_masks.sigmoid()
        intersection = (pred_masks * gt_masks).sum(dim=-1)
        union = pred_masks.sum(dim=-1) + gt_masks.sum(dim=-1) - intersection
        iou = intersection / (union + self.eps)

        # Calculate the smallest convex shape that contains both masks
        pred_coords = pred_masks.nonzero()
        gt_coords = gt_masks.nonzero()
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return 1 - iou.mean()

        enclosing_area = (
            (torch.max(pred_coords[:, -1]) - torch.min(pred_coords[:, -1])) *
            (torch.max(gt_coords[:, -1]) - torch.min(gt_coords[:, -1]))
        )
        
        giou = iou - (enclosing_area - union) / (enclosing_area + self.eps)
        return (1 - giou).mean()

@torch.jit.script
def batch_sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, 
                           alpha: float = 0.25, gamma: float = 2) -> torch.Tensor:
    """Enhanced focal loss with auto-balancing."""
    N = inputs.shape[1]
    
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = ((1 - p_t) ** gamma)
    
    # Auto-balancing
    num_pos = (targets > 0.5).float().sum()
    num_neg = (targets <= 0.5).float().sum()
    pos_weight = torch.where(num_pos > 0, num_neg / (num_pos + num_neg), torch.tensor(0.5).to(inputs.device))
    
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = focal_weight * alpha_t
    
    loss = focal_weight * ce_loss
    return loss.mean() / N

@torch.jit.script
def batch_soft_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, 
                        smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss with smoothing."""
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1) + smooth
    denominator = inputs.sum(-1) + targets.sum(-1) + smooth
    loss = 1 - numerator / denominator
    return loss.mean()

class EnhancedHungarianMatcher(nn.Module):
    """Enhanced Hungarian Matcher with multiple cost metrics."""
    
    def __init__(self, cost_weight: List[float], num_points_thresh: int = 10):
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))
        self.num_points_thresh = num_points_thresh
        self.lovasz_loss = LovaszHingeLoss()
        self.giou_loss = GIoULoss()

    @torch.no_grad()
    def forward(self, pred_labels: torch.Tensor, pred_masks: List[torch.Tensor], 
                insts: List) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
                
            # Classification cost
            pred_label = pred_label.softmax(-1)
            tgt_idx = inst.gt_labels
            cost_class = -pred_label[:, tgt_idx]

            # Get target masks
            tgt_mask = inst.gt_spmasks.float()
            
            # Multiple mask costs
            cost_bce = batch_sigmoid_focal_loss(pred_mask, tgt_mask)
            cost_dice = batch_soft_dice_loss(pred_mask, tgt_mask)
            cost_giou = self.giou_loss(pred_mask.sigmoid(), tgt_mask)
            cost_lovasz = self.lovasz_loss(pred_mask.sigmoid(), (tgt_mask > 0.5).float())

            # Combine costs with weights
            C = (self.cost_weight[0] * cost_class + 
                 self.cost_weight[1] * cost_bce +
                 self.cost_weight[2] * cost_dice +
                 self.cost_weight[3] * cost_giou +
                 self.cost_weight[4] * cost_lovasz)

            # Filter by number of points
            mask_points = tgt_mask.sum(-1)
            valid_mask = mask_points >= self.num_points_thresh
            if valid_mask.any():
                C = C[:, valid_mask]
                indices.append(linear_sum_assignment(C.cpu()))
            else:
                indices.append(([], []))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]

@gorilla.LOSSES.register_module()
class EnhancedCriterion(nn.Module):
    def __init__(
        self,
        ignore_label: int = -100,
        loss_weight: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        cost_weight: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        non_object_weight: float = 0.1,
        num_class: int = 18,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.5,
        num_points_thresh: int = 10,
    ):
        super().__init__()
        
        # Class weights with non-object balancing
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer('class_weight', class_weight)
        
        # Loss weights
        self.loss_weight = nn.Parameter(torch.ones(5), requires_grad=True)
        
        # Initialize matchers and losses
        self.matcher = EnhancedHungarianMatcher(cost_weight, num_points_thresh)
        self.lovasz_loss = LovaszHingeLoss()
        self.giou_loss = GIoULoss()
        
        # Other parameters
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _compute_mask_losses(self, pred_mask: torch.Tensor, tgt_mask: torch.Tensor, 
                           num_masks: int) -> dict:
        """Compute multiple mask losses."""
        losses = {}
        if num_masks == 0:
            return {
                'mask_bce': torch.tensor(0.).to(pred_mask.device),
                'mask_dice': torch.tensor(0.).to(pred_mask.device),
                'mask_giou': torch.tensor(0.).to(pred_mask.device),
                'mask_lovasz': torch.tensor(0.).to(pred_mask.device)
            }

        losses['mask_bce'] = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask)
        losses['mask_dice'] = batch_soft_dice_loss(pred_mask, tgt_mask)
        losses['mask_giou'] = self.giou_loss(pred_mask, tgt_mask)
        losses['mask_lovasz'] = self.lovasz_loss(pred_mask.sigmoid(), (tgt_mask > 0.5).float())
        
        return losses

    def forward(self, pred: dict, insts: List) -> Tuple[torch.Tensor, dict]:
        loss_out = {}
        total_loss = 0

        # Get predictions
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # Match predictions with ground truth
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # Classification loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(pred_labels.shape[:2], self.num_class,
                             dtype=torch.int64, device=pred_labels.device)
        tgt_class[idx] = tgt_class_o
        
        # Enhanced class loss with label smoothing
        class_loss = F.cross_entropy(
            pred_labels.transpose(1, 2), tgt_class,
            self.class_weight, label_smoothing=0.1
        )
        loss_out['cls_loss'] = class_loss.item()
        total_loss += self.loss_weight[0] * class_loss

        # Mask and score losses
        total_masks = 0
        mask_losses = {
            'mask_bce': torch.tensor(0.).to(pred_labels.device),
            'mask_dice': torch.tensor(0.).to(pred_labels.device),
            'mask_giou': torch.tensor(0.).to(pred_labels.device),
            'mask_lovasz': torch.tensor(0.).to(pred_labels.device),
            'score_loss': torch.tensor(0.).to(pred_labels.device)
        }

        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
                
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]
            tgt_mask = inst.gt_spmasks[idx_gt]
            
            # Calculate IoU scores for positive samples
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            # Score loss with quality-aware weighting
            filter_id = torch.where(tgt_score > 0.5)[0]
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                quality_weight = tgt_score.clone().detach()
                mask_losses['score_loss'] += (quality_weight * F.mse_loss(
                    pred_score, tgt_score, reduction='none')).mean()

            # Multiple mask losses
            curr_mask_losses = self._compute_mask_losses(pred_mask, tgt_mask.float(), len(inst))
            for k, v in curr_mask_losses.items():
                mask_losses[k] += v
                
            total_masks += 1

        # Average losses and add to total loss
        if total_masks > 0:
            for k, v in mask_losses.items():
                mask_losses[k] = v / total_masks
                loss_out[k] = mask_losses[k].item()
                if k != 'score_loss':
                    total_loss += self.loss_weight[1] * mask_losses[k]
                else:
                    total_loss += self.loss_weight[2] * mask_losses[k]

        # Auxiliary losses
        if self.use_aux_loss and 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                aux_loss, aux_loss_out = self._compute_aux_loss(i, aux_outputs, insts)
                total_loss += self.aux_loss_weight * aux_loss
                loss_out.update(aux_loss_out)

        loss_out['loss'] = total_loss.item()
        return total_loss, loss_out

    def _compute_aux_loss(self, layer: int, aux_outputs: dict, 
                         insts: List) -> Tuple[torch.Tensor, dict]:
        """Compute auxiliary losses for intermediate outputs."""
        indices = self.matcher(aux_outputs['labels'], aux_outputs['masks'], insts)
        idx = self._get_src_permutation_idx(indices)
        
        # Class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            aux_outputs['labels'].shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=aux_outputs['labels'].device)
        tgt_class[idx] = tgt_class_o

        layer_loss = {}
        
        # Enhanced class loss with label smoothing
        class_loss = F.cross_entropy(
            aux_outputs['labels'].transpose(1, 2),
            tgt_class,
            self.class_weight,
            label_smoothing=0.1
        )
        layer_loss[f'layer_{layer}_cls_loss'] = class_loss.item()

        # Mask and score losses
        total_masks = 0
        mask_losses = {
            'mask_bce': torch.tensor(0.).to(aux_outputs['labels'].device),
            'mask_dice': torch.tensor(0.).to(aux_outputs['labels'].device),
            'mask_giou': torch.tensor(0.).to(aux_outputs['labels'].device),
            'mask_lovasz': torch.tensor(0.).to(aux_outputs['labels'].device),
            'score_loss': torch.tensor(0.).to(aux_outputs['labels'].device)
        }

        for mask, score, inst, (idx_q, idx_gt) in zip(
            aux_outputs['masks'], aux_outputs['scores'], insts, indices):
            if len(inst) == 0:
                continue

            pred_score = score[idx_q]
            pred_mask = mask[idx_q]
            tgt_mask = inst.gt_spmasks[idx_gt]

            # Calculate IoU scores
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            # Score loss with quality-aware weighting
            filter_id = torch.where(tgt_score > 0.5)[0]
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                quality_weight = tgt_score.clone().detach()
                mask_losses['score_loss'] += (quality_weight * F.mse_loss(
                    pred_score, tgt_score, reduction='none')).mean()

            # Multiple mask losses
            curr_mask_losses = self._compute_mask_losses(pred_mask, tgt_mask.float(), len(inst))
            for k, v in curr_mask_losses.items():
                mask_losses[k] += v

            total_masks += 1

        # Average and add to layer losses
        if total_masks > 0:
            for k, v in mask_losses.items():
                mask_losses[k] = v / total_masks
                layer_loss[f'layer_{layer}_{k}'] = mask_losses[k].item()

        # Combine losses
        total_layer_loss = (self.loss_weight[0] * class_loss +
                          self.loss_weight[1] * (mask_losses['mask_bce'] + 
                                               mask_losses['mask_dice'] + 
                                               mask_losses['mask_giou'] + 
                                               mask_losses['mask_lovasz']) / 4 +
                          self.loss_weight[2] * mask_losses['score_loss'])

        return total_layer_loss, layer_loss

def get_iou(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Enhanced IoU calculation with better numeric stability.
    """
    inputs = inputs.sigmoid()
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    
    # Binary prediction with soft threshold
    inputs_binary = (inputs >= 0.5).float()
    targets_binary = (targets > 0.5).float()
    
    # Calculate intersection and union
    intersection = (inputs_binary * targets_binary).sum(-1)
    union = (inputs_binary + targets_binary).sum(-1) - intersection
    
    # IoU with stability term
    iou = (intersection + eps) / (union + eps)
    
    # Additional soft IoU term for better gradients
    soft_intersection = (inputs * targets).sum(-1)
    soft_union = inputs.sum(-1) + targets.sum(-1) - soft_intersection
    soft_iou = (soft_intersection + eps) / (soft_union + eps)
    
    # Combine hard and soft IoU
    combined_iou = 0.8 * iou + 0.2 * soft_iou
    return combined_iou