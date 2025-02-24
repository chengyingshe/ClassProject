import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .structures import InstanceData_
from mmdet3d.registry import MODELS, TASK_UTILS
import torch.nn as nn

def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


@MODELS.register_module()
class InstanceCriterion:
    """Instance criterion.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes (int): number of classes.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum()
            for mask losses.

    """

    def __init__(self, matcher, loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss=False):
        self.matcher = TASK_UTILS.build(matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData_
    def __call__(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Dict: with instance loss value.
        """
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'inst_loss': loss}



@MODELS.register_module()
class InstanceCriterionV2(nn.Module):
    """修改后的损失函数
        - 将原来的平衡权重修改成了可训练的参数
        - 添加边界损失
        - 对于各个损失部分都添加了数值稳定处理，防止其中某一个损失函数对训练的影响较大
    
    Args:
        matcher (Callable): 用于匹配预测和GT的匹配器类
        loss_weight (List[float]): 分类、掩码BCE、掩码DICE、分数损失的权重
        non_object_weight (float): 背景类别的权重
        num_classes (int): 类别数量
        fix_dice_loss_weight (bool): 是否固定DICE损失权重
        iter_matcher (bool): 是否为每个解码器层使用独立匹配器
        fix_mean_loss (bool): 是否使用mean而不是sum来聚合掩码损失
        focal_gamma (float): focal loss的gamma参数
        boundary_weight (float): 边界损失的权重
    """
    def __init__(self, 
                 matcher,
                 non_object_weight,
                 num_classes,
                 fix_dice_loss_weight,
                 iter_matcher,
                 fix_mean_loss=False,
                 focal_gamma=2.0,
                 boundary_weight=1.0, 
                 **kwargs):
        super().__init__()
        self.matcher = TASK_UTILS.build(matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = nn.Parameter(torch.ones(5), requires_grad=True)
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight
        
        self.log_vars = nn.Parameter(torch.zeros(6).clamp(-10, 10), requires_grad=True)

    def get_balanced_loss(self, losses):
        balanced_loss = 0
        for i in range(len(self.loss_weight)):
            balanced_loss += self.loss_weight[i] * losses[i]
        return balanced_loss

    def improved_dice_loss(self, inputs, targets):
        inputs = inputs.sigmoid()
        
        # 添加数值稳定性
        focal_weight = torch.clamp((1 - inputs) ** self.focal_gamma, min=1e-8)
        numerator = 2 * (focal_weight * inputs * targets).sum(-1)
        denominator = (focal_weight * inputs).sum(-1) + targets.sum(-1)

        loss = 1 - (numerator + 1) / (denominator + 1 + 1e-8)
        return loss.mean()

    def compute_boundary_loss(self, pred_mask, gt_mask):
        """计算边界感知损失."""
        
        def get_boundary(mask):
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
                
            mask = mask.float()
                
            laplacian_kernel = torch.tensor([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]], dtype=torch.float32, device=mask.device)
            laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
            
            if mask.shape[1] > 1:
                laplacian_kernel = laplacian_kernel.repeat(mask.shape[1], 1, 1, 1)
                
            pad = nn.functional.pad(mask, (1, 1, 1, 1), mode='replicate')
            boundary = F.conv2d(pad, laplacian_kernel, groups=mask.shape[1])
            return boundary.squeeze()

        
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()

        pred_boundary = get_boundary(pred_mask)
        gt_boundary = get_boundary(gt_mask)

        # 归一化边界强度到[0,1]范围
        pred_boundary = pred_boundary / (pred_boundary.max() + 1e-8)
        gt_boundary = gt_boundary / (gt_boundary.max() + 1e-8)
        
        # 使用平衡权重
        pos_weight = (gt_boundary < 0.5).float().sum() / (gt_boundary >= 0.5).float().sum().clamp(min=1e-8)
        boundary_weight = torch.where(gt_boundary >= 0.5, 
                                    pos_weight * torch.ones_like(gt_boundary),
                                    torch.ones_like(gt_boundary))
        
        # 添加L1损失来稳定训练
        boundary_l1 = F.l1_loss(pred_boundary, gt_boundary)
        boundary_bce = F.binary_cross_entropy_with_logits(
            pred_boundary, gt_boundary, 
            weight=boundary_weight)
            
        return boundary_bce + 0.5 * boundary_l1  # 结合BCE和L1损失

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """计算中间层的损失."""
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        losses = self._compute_losses(cls_preds, pred_scores, pred_masks, insts, indices)
        loss = self.get_balanced_loss(list(losses.values()))
        return loss

    def _compute_losses(self, cls_preds, pred_scores, pred_masks, insts, indices):
        """计算所有损失项."""
        # 分类损失
        cls_losses = []
        
        # cls_loss
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses, boundary_losses = [], [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                     insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            
            # mask_bce_loss
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            
            # mask_dice_loss (improved)
            mask_dice_losses.append(self.improved_dice_loss(pred_mask, tgt_mask.float()))
            
            # boundary_loss (new part)
            boundary_losses.append(self.compute_boundary_loss(pred_mask, tgt_mask))

            if score is not None:
                pred_score = score[idx_q]
                with torch.no_grad():
                    tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

                filter_id, _ = torch.where(tgt_score > 0.5)
                if filter_id.numel():
                    tgt_score = tgt_score[filter_id]
                    pred_score = pred_score[filter_id]
                    score_losses.append(F.mse_loss(pred_score, tgt_score))
        
        # 聚合损失
        losses = {}
        losses['cls_loss'] = cls_loss
        
        if len(mask_bce_losses):
            if self.fix_mean_loss:
                mask_bce_loss = torch.stack(mask_bce_losses).mean()
                mask_dice_loss = torch.stack(mask_dice_losses).mean()
            else:
                mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
                mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)
            
            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0
            
        losses['mask_bce_loss'] = mask_bce_loss
        losses['mask_dice_loss'] = mask_dice_loss
        
        if boundary_losses:
            boundary_loss = torch.stack(boundary_losses).mean()
            boundary_loss = boundary_loss * self.boundary_weight
        else:
            boundary_loss = 0
            
        losses['boundary_loss'] = boundary_loss
        
        if score_losses:
            score_loss = torch.stack(score_losses).mean()
        else:
            score_loss = 0
            
        losses['score_loss'] = score_loss
        
        return losses

    def forward(self, pred, insts):
        """主损失计算函数."""
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # 匹配预测和GT
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        # 计算所有损失项
        losses = self._compute_losses(cls_preds, pred_scores, pred_masks, insts, indices)
        # print('####losses:', losses)
        # print('####loss_weight:', self.loss_weight)
        # 动态平衡损失
        total_loss = self.get_balanced_loss(list(losses.values()))
        # print('####total_loss:', total_loss)

        # 添加辅助损失
        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None 
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                total_loss += self.get_layer_loss(aux_outputs, insts, indices)
        # print('####total_loss2:', total_loss)
        
        return {'inst_loss': total_loss}


@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@TASK_UTILS.register_module()
class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """
    def __init__(self, costs):
        self.costs = []
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        cost_value = torch.stack(cost_values).sum(dim=0)
        query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())
        return labels.new_tensor(query_ids), labels.new_tensor(object_ids)


@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]


@MODELS.register_module()
class OneDataCriterion:
    """Loss module for SPFormer.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
    """

    def __init__(self, matcher, loss_weight, non_object_weight, 
                 num_classes_1dataset, num_classes_2dataset,
                 fix_dice_loss_weight, iter_matcher):
        self.matcher = TASK_UTILS.build(matcher)
        self.num_classes_1dataset = num_classes_1dataset
        self.num_classes_2dataset = num_classes_2dataset
        self.class_weight_1dataset = [1] * num_classes_1dataset + [non_object_weight]
        self.class_weight_2dataset = [1] * num_classes_2dataset + [non_object_weight]
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            if cls_pred.shape[1] == self.num_classes_1dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_1dataset)))
            elif cls_pred.shape[1] == self.num_classes_2dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_2dataset)))
            else:
                raise RuntimeError(
                    f'Invalid classes number {cls_pred.shape[1]}.')

        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
        mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData
    def __call__(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_gts, n_classes + 1);
                List `scores` of len batch_size each of shape (n_gts, 1);
                List `masks` of len batch_size each of shape (n_gts, n_points).
                Dict `aux_preds` with list of cls_preds, scores, and masks.
        """
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]            
            if cls_pred.shape[1] == self.num_classes_1dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_1dataset)))
            elif cls_pred.shape[1] == self.num_classes_2dataset + 1:
                cls_losses.append(F.cross_entropy(
                    cls_pred, cls_target,
                    cls_pred.new_tensor(self.class_weight_2dataset)))
            else:
                raise RuntimeError(
                    f'Invalid classes number {cls_pred.shape[1]}.')
        
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
        mask_dice_loss = torch.stack(mask_dice_losses).sum()

        if self.fix_dice_loss_weight:
            mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'inst_loss': loss}
