import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MultiScaleFusion(nn.Module):
    def __init__(self, d_model: int, num_scales: int = 4):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.scale_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.scale_weights, dim=0)
        normalized_features = [norm(feat) for norm, feat in zip(self.scale_norms, features_list)]
        fused_features = sum(w * feat for w, feat in zip(weights, normalized_features))
        return self.fusion_layer(fused_features)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Two-layer feed-forward network
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()

    def forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x2)

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, source: torch.Tensor, query: torch.Tensor, batch_offsets: List[int], 
                attn_masks: Optional[List[torch.Tensor]] = None, pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)

        for i in range(B):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)
            q = query[i].unsqueeze(0)
            
            # Self attention
            if attn_masks:
                attn_output, _ = self.multihead_attn(q, k, v, attn_mask=attn_masks[i])
            else:
                attn_output, _ = self.multihead_attn(q, k, v)
            
            # Add & Norm
            q = q + self.dropout1(attn_output)
            q = self.norm1(q)
            
            # FFN
            q2 = self.forward_ffn(q)
            q = q + q2
            q = self.norm2(q)
            
            outputs.append(q)
        
        return torch.cat(outputs, dim=0)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()

    def with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x2)

    def forward(self, x: torch.Tensor, pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = k = self.with_pos_embed(x, pe)
        v = x
        
        # Self attention
        attn_output, _ = self.multihead_attn(q, k, v)
        
        # Add & Norm
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # FFN
        x2 = self.forward_ffn(x)
        x = x + x2
        x = self.norm2(x)
        
        return x


class QueryDecoder(nn.Module):
    def __init__(
        self,
        num_layer: int = 6,
        num_query: int = 100,
        num_class: int = 18,
        in_channel: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
        activation_fn: str = 'gelu',
        iter_pred: bool = False,
        attn_mask: bool = False,
        pe: bool = True,
        use_multiscale: bool = True
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.d_model = d_model
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.use_multiscale = use_multiscale

        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(in_channel, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Query embeddings
        self.query = nn.Parameter(torch.zeros(num_query, d_model))
        nn.init.xavier_uniform_(self.query)
        self.query_norm = nn.LayerNorm(d_model)

        # Positional encodings
        if pe:
            self.pe = nn.Parameter(torch.zeros(num_query, d_model))
            nn.init.xavier_uniform_(self.pe)

        # Multi-scale fusion
        if use_multiscale:
            self.multiscale_fusion = MultiScaleFusion(d_model)

        # Cross and self attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dropout) for _ in range(num_layer)
        ])
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(d_model, nhead, dropout) for _ in range(num_layer)
        ])

        # Prediction heads
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_class + 1)
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid()
        )

        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.out_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_mask(self, query: torch.Tensor, mask_feats: torch.Tensor, 
                 batch_offsets: List[int]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        pred_masks = []
        attn_masks = []
        
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            
            # Enhanced mask prediction with non-linear activation
            pred_mask = torch.einsum('nd,md->nm', self.mask_head(query[i]), mask_feat)
            
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def prediction_head(self, query: torch.Tensor, mask_feats: torch.Tensor, 
                       batch_offsets: List[int]) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        query = self.out_norm(query)
        pred_labels = self.class_head(query)
        pred_scores = self.score_head(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x: torch.Tensor, batch_offsets: List[int]) -> dict:
        # Feature projection
        inst_feats = self.input_proj(x)
        mask_feats = self.mask_head(inst_feats)

        # Initialize queries
        B = len(batch_offsets) - 1
        query = self.query_norm(self.query).unsqueeze(0).expand(B, -1, -1)
        pe = getattr(self, 'pe', None)

        # Multi-layer processing
        layer_features = []
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets)
            query = self.self_attn_layers[i](query, pe)
            layer_features.append(query)

        # Multi-scale fusion if enabled
        if self.use_multiscale and len(layer_features) > 1:
            query = self.multiscale_fusion(layer_features)

        # Generate predictions
        pred_labels, pred_scores, pred_masks, _ = self.prediction_head(query, mask_feats, batch_offsets)
        
        return {
            'labels': pred_labels,
            'masks': pred_masks,
            'scores': pred_scores
        }

    def forward_iter_pred(self, x: torch.Tensor, batch_offsets: List[int]) -> dict:
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []

        # Feature projection
        inst_feats = self.input_proj(x)
        mask_feats = self.mask_head(inst_feats)

        # Initialize queries
        B = len(batch_offsets) - 1
        query = self.query_norm(self.query).unsqueeze(0).expand(B, -1, -1)
        pe = getattr(self, 'pe', None)

        # Initial predictions
        pred_labels, pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_offsets)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        # Iterative refinement
        layer_features = []
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets, attn_masks, pe)
            query = self.self_attn_layers[i](query, pe)
            layer_features.append(query)

            # Multi-scale fusion if enabled
            if self.use_multiscale and len(layer_features) > 1:
                fused_query = self.multiscale_fusion(layer_features)
            else:
                fused_query = query

            pred_labels, pred_scores, pred_masks, attn_masks = self.prediction_head(fused_query, mask_feats, batch_offsets)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)

        return {
            'labels': pred_labels,
            'masks': pred_masks,
            'scores': pred_scores,
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c
            } for a, b, c in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
        }

    def forward(self, x: torch.Tensor, batch_offsets: List[int]) -> dict:
        if self.iter_pred:
            return self.forward_iter_pred(x, batch_offsets)
        else:
            return self.forward_simple(x, batch_offsets)