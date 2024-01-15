# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from typing import Optional
from torch import nn, Tensor
from torch.nn import functional as F
from mapmaster.models.utils.misc import Conv2d, c2_xavier_fill, get_activation_fn
from mapmaster.models.utils.position_encoding import PositionEmbeddingSine


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PointMask2TransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            num_feature_levels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            query_split=(20, 25, 15),
            max_pieces=(10, 2, 30), 
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        # d_model, nhead, dropout = 0.0, activation = "relu", normalize_before = False
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.max_pieces = max_pieces
        self.query_split = query_split
        self.num_queries = sum([self.max_pieces[i] * self.query_split[i] for i in range(len(self.max_pieces))])

        # learnable pt features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim).weight      # [700, C]
        # learnable pt p.e.
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim).weight     # [20 * 10 + 25 * 2 + 15 * 30, C] which is [700, C], note the memory

        # level embedding (we always use 3 scales)
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.ModuleList(
                nn.Linear(hidden_dim*self.max_pieces[i], num_classes + 1) for i in range(len(query_split))
                ) 
        self.mask_embed = nn.ModuleList(
            MLP(hidden_dim*self.max_pieces[i], hidden_dim, mask_dim, 3) for i in range(len(query_split))
            )

        # split info
        self.cls_split = [self.query_split[i] * self.max_pieces[i] for i in range(len(self.query_split))]
        self.ins_split = torch.cat([torch.ones(self.query_split[i], dtype=torch.long)*self.max_pieces[i] for i in range(len(self.query_split))])

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            mask = torch.zeros((x[i].size(0), x[i].size(2), x[i].size(3)), device=x[i].device, dtype=torch.bool)
            pos.append(self.pe_layer(mask).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.unsqueeze(1).repeat(1, bs, 1)      # [n_q, bs, C]
        output = self.query_feat.unsqueeze(1).repeat(1, bs, 1)            # [n_q, bs, C]

        predictions_class = []
        predictions_mask = []
        decoder_outputs = []

        # prediction heads on learnable query features
        dec_out, outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        decoder_outputs.append(dec_out)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](output)
            dec_out, outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            decoder_outputs.append(dec_out)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class,
            'pred_masks': predictions_mask,
            'decoder_outputs': decoder_outputs
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)   # (b, q, c')  (b, 700, c')
        decoder_output_split = decoder_output.split(self.cls_split, dim=1) 

        outputs_class, mask_embed = None, None
        for i in range(len(self.query_split)):
            x_cls_pt = decoder_output_split[i]                                   # (bs, n_q*n_pt, c')
            n_q, n_pt = self.query_split[i], self.max_pieces[i]
            x_cls_pt = x_cls_pt.reshape(x_cls_pt.shape[0], n_q, -1)        # (bs, n_q, n_pt * c')
            x_cls = self.class_embed[i](x_cls_pt)                             # (bs, n_q, 2)
            x_cls_ins = self.mask_embed[i](x_cls_pt)                          # （bs, n_q, c)
            if outputs_class is None:
                outputs_class = x_cls
                mask_embed = x_cls_ins
            else:
                outputs_class = torch.cat([outputs_class, x_cls], dim=1)
                mask_embed = torch.cat([mask_embed, x_cls_ins], dim=1)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)    # for loss cal

        outputs_mask_split = outputs_mask.split(self.query_split, dim=1)

        attn_mask = []
        for i in range(len(self.query_split)):
            # [bs, n_q, h, w] -> [bs, n_q, 1, h, w] -> [bs, n_q, n_pt, h, w] -> [bs, n_q*n_pt, h, w]
            attn_mask.append(outputs_mask_split[i].unsqueeze(2).repeat(1, 1, self.max_pieces[i], 1, 1).flatten(1, 2))
        attn_mask = torch.cat(attn_mask, dim=1)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return decoder_output, outputs_class, outputs_mask, attn_mask
