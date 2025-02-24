import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from spformer.utils import cuda_cast, rle_encode
from .backbone_v2 import ResidualBlock, UBlock
from .loss_v2 import EnhancedCriterion
from .loss import Criterion
from .query_decoder_v2 import QueryDecoder


class AdaptiveWeightModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LayerNorm(channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 3),  # 3 weights for mean, max, sum
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avgpool(x.unsqueeze(-1)).squeeze(-1)
        weights = self.fc(y)  # [B, 3]
        return weights


class EnhancedFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive weight module
        self.weight_module = AdaptiveWeightModule(out_channels)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x, superpoints):
        # Transform features
        x = self.transform(x)
        
        # Get multiple pooled features
        mean_feat = scatter_mean(x, superpoints, dim=0)
        max_feat = scatter_max(x, superpoints, dim=0)[0]
        sum_feat = scatter_sum(x, superpoints, dim=0)
        
        # Get adaptive weights
        weights = self.weight_module(mean_feat)  # [B, 3]
        
        # Weighted combination
        pooled_feat = (weights[:, 0:1] * mean_feat + 
                      weights[:, 1:2] * max_feat + 
                      weights[:, 2:3] * sum_feat)
        
        # Apply channel attention
        channel_weights = self.channel_attention(pooled_feat)
        pooled_feat = pooled_feat * channel_weights
        
        # Final projection
        out_feat = self.output_proj(pooled_feat)
        return out_feat


@gorilla.MODELS.register_module()
class SPFormer(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before: bool = True,
        return_blocks: bool = True,
        pool: str = 'enhanced',  # Changed default pooling to enhanced
        num_class: int = 18,
        decoder: dict = None,
        criterion: dict = None,
        test_cfg: dict = None,
        norm_eval: bool = False,
        fix_module: list = [],
        use_instance_norm: bool = True,  # Added instance normalization option
        use_droppath: bool = True,  # Added drop path option
        droppath_rate: float = 0.1,
    ):
        super().__init__()
        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.use_instance_norm = use_instance_norm
        
        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_channel, media),
            nn.LayerNorm(media) if not use_instance_norm else nn.InstanceNorm1d(media),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Sparse convolution encoder
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
            
        if use_instance_norm:
            self.input_norm = nn.InstanceNorm1d(media)
        else:
            self.input_norm = nn.BatchNorm1d(media, eps=1e-4, momentum=0.1)
            
        self.input_act = nn.ReLU(inplace=True)

        # Build backbone
        block = ResidualBlock
        norm_fn = functools.partial(
            nn.InstanceNorm1d if use_instance_norm else nn.BatchNorm1d,
            eps=1e-4, momentum=0.1
        )
        
        # Progressive channel growth
        block_list = [media * (2 ** i) for i in range(blocks)]
        
        # UBlock with drop path
        if use_droppath:
            self.droppath = nn.ModuleList([
                nn.Dropout(p=droppath_rate * i / (blocks - 1)) 
                for i in range(blocks)
            ])
        
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        
        # Enhanced output processing
        self.output_layer = spconv.SparseSequential(
            norm_fn(media),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(media, media, kernel_size=3, padding=1, bias=False)
        )
        
        # Feature aggregation
        if pool == 'enhanced':
            self.pool_layer = EnhancedFeatureAggregation(media, media)
        self.pool = pool
        self.num_class = num_class

        # Decoder setup
        decoder_cfg = {
            'in_channel': media,
            'num_class': num_class,
            **decoder
        }
        self.decoder = QueryDecoder(**decoder_cfg)

        # Loss criterion
        criterion_cfg = {
            'num_class': num_class,
            **criterion
        }
        self.criterion = EnhancedCriterion(**criterion_cfg)
        # self.criterion = Criterion(**criterion_cfg)
        
        # Fix modules if specified
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.InstanceNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, 
             feats, insts, superpoints, batch_offsets):
        batch_size = len(batch_offsets) - 1
        
        # Enhanced voxelization
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        voxel_feats = F.normalize(voxel_feats, p=2, dim=1)
        
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), 
                                      spatial_shape, batch_size)
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        
        # Decoder predictions
        out = self.decoder(sp_feats, batch_offsets)
        
        # Loss computation
        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape,
                feats, insts, superpoints, batch_offsets):
        batch_size = len(batch_offsets) - 1
        
        # Feature extraction
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), 
                                      spatial_shape, batch_size)
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        
        # Get predictions
        out = self.decoder(sp_feats, batch_offsets)
        
        # Post-process predictions
        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']
        pred_scores = out['scores']

        # Enhanced prediction processing
        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= F.sigmoid(pred_scores[0])  # Use sigmoid activation
        
        # Generate class labels
        labels = torch.arange(
            self.num_class, device=scores.device).unsqueeze(0).repeat(
            self.decoder.num_query, 1).flatten(0, 1)
        
        # Instance selection with NMS-like process
        scores_sorted, indices = scores.flatten(0, 1).sort(descending=True)
        selected_indices = []
        selected_masks = []
        
        for idx in indices:
            if len(selected_indices) >= self.test_cfg.topk_insts:
                break
                
            current_mask = pred_masks[0][idx // self.num_class]
            
            # Check overlap with previously selected masks
            overlap = False
            for selected_mask in selected_masks:
                iou = (current_mask * selected_mask).sum() / (
                    (current_mask + selected_mask) > 0).sum()
                if iou > 0.3:  # IoU threshold
                    overlap = True
                    break
            
            if not overlap:
                selected_indices.append(idx)
                selected_masks.append(current_mask)
        
        # Get final predictions
        topk_indices = torch.tensor(selected_indices, device=scores.device)
        scores = scores.flatten(0, 1)[topk_indices]
        labels = labels[topk_indices]
        labels += 1
        
        mask_indices = torch.div(topk_indices, self.num_class, rounding_mode='floor')
        mask_pred = pred_masks[0][mask_indices]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred = (mask_pred > 0).float()
        
        # Compute mask scores with IoU term
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        
        # Apply to superpoints
        mask_pred = mask_pred[:, superpoints].int()

        # Apply thresholds
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        # Convert to numpy
        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        # Generate predictions
        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {
                'scan_id': scan_ids[0],
                'label_id': cls_pred[i],
                'conf': score_pred[i],
                'pred_mask': rle_encode(mask_pred[i])
            }
            pred_instances.append(pred)

        return dict(
            scan_id=scan_ids[0],
            pred_instances=pred_instances,
            gt_instances=insts[0].gt_instances
        )

    def extract_feat(self, x, superpoints, p2v_map):
        # Initial feature processing
        x = self.input_conv(x)
        x.features = self.input_norm(x.features)
        x.features = self.input_act(x.features)
        
        # Backbone feature extraction with drop path
        if hasattr(self, 'droppath'):
            feat_list = []
            for i, (conv, drop) in enumerate(zip(self.unet.convs, self.droppath)):
                x = conv(x)
                if self.training:
                    x = drop(x)
                feat_list.append(x)
            x = feat_list[-1]
        else:
            x, _ = self.unet(x)
            
        # Output processing
        x = self.output_layer(x)
        x = x.features[p2v_map.long()]

        # Feature aggregation
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)
        elif self.pool == 'enhanced':
            x = self.pool_layer(x, superpoints)
            
        return x