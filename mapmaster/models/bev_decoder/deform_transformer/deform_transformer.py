import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .ops import MSDeformAttn
from .position_encoding import PositionEmbeddingSine
from .position_encoding import PositionEmbeddingLearned

class DeformTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        src_shape=(16, 168),
        tgt_shape=(32, 32),
        d_model=256,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        dec_n_points=4,
        enc_n_points=4,
        src_pos_encode="sine",
        tgt_pos_encode="learned",
        norm_layer=nn.BatchNorm2d,
        use_checkpoint=False,
        use_projection=False,
        map_size=(400, 200),
        image_shape=(900, 1600),
        map_resolution=0.15,
        image_order=(2, 1, 0, 5, 4, 3)
    ):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if isinstance(src_shape[0], int):
            src_shape = [src_shape]
        assert len(src_shape) == len(in_channels)
        n_levels = len(in_channels)

        self.input_proj = nn.ModuleList()
        for i in range(len(in_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], d_model, kernel_size=1, bias=False),
                    norm_layer(d_model),
                )
            )

        encoder_layer = DeformTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, n_levels, n_heads, enc_n_points, use_checkpoint
        )
        self.encoder = DeformTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, n_levels, n_heads, dec_n_points, use_checkpoint
        )
        self.decoder = DeformTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.dropout = nn.Dropout(dropout)

        self.t2s_reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

        if src_pos_encode == "sine":
            self.src_pos_embed = PositionEmbeddingSine(d_model, normalize=True)
            self.src_lvl_embed = nn.Embedding(n_levels, d_model)
        elif src_pos_encode == "learned":
            self.src_pos_embed = nn.ModuleList(
                [PositionEmbeddingLearned(shape, d_model) for shape in src_shape],
            )
        else:
            raise NotImplementedError

        if tgt_pos_encode == "sine":
            self.tgt_pos_embed = PositionEmbeddingSine(d_model, normalize=True)
        elif tgt_pos_encode == "learned":
            self.tgt_pos_embed = PositionEmbeddingLearned(tgt_shape, d_model)
        else:
            raise NotImplementedError

        self.tgt_embed = PositionEmbeddingLearned(tgt_shape, d_model)

        self.src_shape = src_shape
        self.tgt_shape = tgt_shape
        self.src_pos_encode = src_pos_encode
        self.tgt_pos_encode = tgt_pos_encode

        """
        use_projection: bool / whether to use IPM as the reference points 
        map_size: (x_width, y_width) shape of the original Map (400, 200)
        image_shape: (Height, Width)
        map_resolution: map resolution (m / pixel)
        """
        self.use_projection = use_projection  # Use IPM Projection to get reference points
        self.map_size = map_size
        self.map_resolution = map_resolution
        self.image_shape = image_shape
        image_order = torch.tensor(image_order, dtype=torch.long)
        self.register_buffer("image_order", image_order)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.xavier_uniform_(self.t2s_reference_points.weight, gain=1.0)
        nn.init.constant_(self.t2s_reference_points.bias, 0.0)

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_projection_points(self, extrinsic, intrinsic, flip=False):
        """
        extrinsic:
            torch.Tensor (6, 4, 4)
        intrinsic:
            torch.Tensor (6, 3, 3)
        flip:
            flip or not

        Return
            reference points (N, L, 2)
            mask (N, )
        """
        map_forward_ratio = self.tgt_shape[0] / self.map_size[0]  
        map_lateral_ratio = self.tgt_shape[1] / self.map_size[1]

        map_forward_res = self.map_resolution / map_forward_ratio
        map_lateral_res = self.map_resolution / map_lateral_ratio

        X = (torch.arange(self.tgt_shape[0] - 1, -1, -1, device=extrinsic.device) + 0.5 - self.tgt_shape[0] / 2) * map_forward_res
        Y = (torch.arange(self.tgt_shape[1] - 1, -1, -1, device=extrinsic.device) + 0.5 - self.tgt_shape[1] / 2) * map_lateral_res
        if flip:
            Y = -1 * Y  # Flip the Y axis

        Z = torch.zeros(self.tgt_shape, device=extrinsic.device)
        grid_X, grid_Y = torch.meshgrid(X, Y)
        coords = torch.stack([grid_X, grid_Y, Z, torch.ones(self.tgt_shape, device=extrinsic.device)], dim=-1)  # (H, W, 4) homogeneous coordinates
        coords_flatten = coords.reshape(-1, 4)  # (N, 4)

        cams = []
        for cam in extrinsic:
            cam_coords = torch.linalg.inv(cam) @ coords_flatten.T  # (4, N)
            cam_coords = cam_coords[:3, :]  # (3, N) -- x, y, z
            cams.append(cam_coords)
        cams = torch.stack(cams, dim=0)  # (6, 3, N) Coordinates in Camera Frame
        normed_coors = F.normalize(cams, p=1, dim=0)  # (6, 3, N) Normalized Coordinates in Camera Frame

        cams_z = normed_coors[:, 2, :]  # (6, N) z coord
        cam_id = torch.argmax(cams_z, dim=0)  # (N, ) -- bev to img idx, Choose the camera with the smallest angle of view

        max_z = cams_z[cam_id, torch.arange(cams.shape[-1])]
        valid_mask = max_z > 0

        intrinsic_percam = intrinsic[cam_id]  # (N, 3, 3)

        coords_percam = cams[cam_id, :, torch.arange(cams.shape[2])]  # (N, 3)
        pixel_coord = (intrinsic_percam @ coords_percam[:, :, None]).squeeze()  # (N, 3)
        pixel_coord = pixel_coord[:, :2] / pixel_coord[:, [2]]  # divided by Z / (N, 2)

        if not isinstance(self.image_shape, list):
            image_shape = torch.tensor([self.image_shape for _ in range(len(extrinsic))], device=extrinsic.device)[cam_id]
        else:
            image_shape = torch.tensor(self.image_shape, device=extrinsic.device)[cam_id]

        valid_pixelx = torch.bitwise_and(pixel_coord[:, 0] < image_shape[:,1], pixel_coord[:, 0] >= 0)
        valid_pixely = torch.bitwise_and(pixel_coord[:, 1] < image_shape[:,0], pixel_coord[:, 1] >= 0)
        valid_mask = valid_mask * valid_pixelx * valid_pixely

        # cast to levels
        reference_points = []
        for level_shape in self.src_shape:
            level_h, level_w = level_shape
            level_w /= 6  
            image_h, image_w = image_shape.T

            ratio_h = image_h / level_h
            ratio_w = image_w / level_w

            if flip:
                img_x = level_w - pixel_coord[:, 0] / ratio_w
                cam_id_ = self.image_order[cam_id]
                x = cam_id_ * level_w + img_x

            else:
                x = cam_id * level_w + pixel_coord[:, 0] / ratio_w

            y = pixel_coord[:, 1] / ratio_h

            x /= (level_w * 6)  # Normalize to [0 ~ 1]
            y /= level_h

            level_point = torch.stack([x, y], dim=-1)
            reference_points.append(level_point)

        reference_points = torch.stack(reference_points, dim=-2)
        return reference_points, valid_mask

    def forward(self, srcs, src_masks=None, cameras_info=None):

        if not isinstance(srcs, (list, tuple)):
            srcs = [srcs]
        if not isinstance(src_masks, (list, tuple)):
            src_masks = [src_masks]

        if src_masks[0] is None:
            src_masks = []

        src_flatten = []
        src_mask_flatten = []
        src_pos_embed_flatten = []
        src_spatial_shapes = []
        for i, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            src_spatial_shapes.append(spatial_shape)

            if len(src_masks) < i + 1:
                src_mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)  # (N, H, W)
                src_masks.append(src_mask)
            else:
                src_mask = src_masks[i]

            if self.src_pos_encode == "sine":
                src_pos_embed = self.src_pos_embed(src_mask)  # (N, C, H, W)
                src_pos_embed = src_pos_embed + self.src_lvl_embed.weight[i].view(-1, 1, 1)  # (N, C, H, W)
            else:
                src_pos_embed = self.src_pos_embed[i](src_mask)  # (N, C, H, W)

            src = self.input_proj[i](src)  # (N, C, H, W)
            src = src + src_pos_embed  # (N, C, H, W)

            src = src.flatten(2).transpose(1, 2)  # (N, H * W, C)
            src_mask = src_mask.flatten(1)  # (N, H * W)
            src_pos_embed = src_pos_embed.flatten(2).transpose(1, 2)  # (N, H * W, C)

            src_flatten.append(src)
            src_mask_flatten.append(src_mask)
            src_pos_embed_flatten.append(src_pos_embed)

        src = torch.cat(src_flatten, 1)  # (N, L * H * W, C)
        src_mask = torch.cat(src_mask_flatten, 1)  # (N, L * H * W)
        src_pos_embed = torch.cat(src_pos_embed_flatten, 1)  # (N, L * H * W, C)
        src_spatial_shapes = torch.as_tensor(src_spatial_shapes, dtype=torch.long, device=src.device)  # (L, 2)
        src_level_start_index = torch.cat(
            (src_spatial_shapes.new_zeros((1,)), src_spatial_shapes.prod(1).cumsum(0)[:-1])
        )  # (L,)
        src_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in src_masks], 1)  # (N, L, 2)

        tgt_mask = torch.zeros((srcs[0].size(0), *self.tgt_shape), dtype=torch.bool, device=srcs[0].device)  # (N, H, W)
        tgt_pos_embed = self.tgt_pos_embed(tgt_mask)  # (N, C, H, W)
        tgt_pos_embed = tgt_pos_embed.flatten(2).transpose(1, 2)  # (N, H * W, C)
        # tgt = tgt_pos_embed  # (N, H * W, C)
        tgt = self.tgt_embed(tgt_mask).flatten(2).transpose(1, 2)

        tgt_spatial_shapes = torch.as_tensor(self.tgt_shape, dtype=torch.long, device=tgt.device).unsqueeze(0)  # (1, 2)
        tgt_valid_ratios = self.get_valid_ratio(tgt_mask).unsqueeze(1)  # (N, 1, 2)
        tgt_level_start_index = tgt_spatial_shapes.new_zeros((1,))  # (1,)
        tgt_mask = tgt_mask.flatten(1)  # (N, 1 * H * W)

        t2s_reference_points = self.t2s_reference_points(tgt_pos_embed).sigmoid()  # (N, H * W, 2)

        if self.use_projection:
            t2s_reference_points = t2s_reference_points.unsqueeze(-2).repeat(1, 1, len(self.src_shape), 1)  # (N, H * W, L, 2)
            bs = srcs[0].shape[0]

            do_flip = cameras_info['do_flip']
            if do_flip is None:
                do_flip = torch.zeros((bs,), dtype=torch.bool)

            for i in range(bs):
                flip = do_flip[i].item()
                extrinsic = cameras_info['extrinsic'][i].float()
                intrinsic = cameras_info['intrinsic'][i].float()

                # Use IPM to generate reference points, Original Size (900, 1600)
                ipm_reference_points, valid_mask = self.get_projection_points(extrinsic, intrinsic, flip)  # (N, L, 2)
                loc = torch.where(valid_mask > 0)[0]

                # Change the embeddings to reference point coordinate
                t2s_reference_points[i, loc, :, :] = ipm_reference_points[loc, :, :]
        else:
            t2s_reference_points = t2s_reference_points[:, :, None]

        # encoder
        memory = self.encoder(
            src, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_pos_embed, src_mask
        )  # (N, H * W, C)
        # decoder
        hs = self.decoder(
            tgt,
            memory,
            tgt_pos_embed,
            t2s_reference_points,
            tgt_spatial_shapes,
            src_spatial_shapes,
            tgt_level_start_index,
            src_level_start_index,
            tgt_valid_ratios,
            src_valid_ratios,
            tgt_mask,
            src_mask,
        )  # (M, N, H * W, C)
        ys = hs.transpose(2, 3)  # (M, N, C, H * W)
        ys = ys.reshape(*ys.shape[:-1], *self.tgt_shape).contiguous()  # (M, N, C, H, W)
        return [memory, hs, ys]


class DeformTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_checkpoint=False,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.use_checkpoint = use_checkpoint

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def _forward(
        self, src, src_pos_embed, src_reference_points, src_spatial_shapes, src_level_start_index, src_key_padding_mask
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, src_pos_embed),
            src_reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src

    def forward(
        self, src, src_pos_embed, src_reference_points, src_spatial_shapes, src_level_start_index, src_key_padding_mask
    ):
        if self.use_checkpoint and self.training:
            src = checkpoint.checkpoint(
                self._forward,
                src,
                src_pos_embed,
                src_reference_points,
                src_spatial_shapes,
                src_level_start_index,
                src_key_padding_mask,
            )
        else:
            src = self._forward(
                src,
                src_pos_embed,
                src_reference_points,
                src_spatial_shapes,
                src_level_start_index,
                src_key_padding_mask,
            )
        return src


class DeformTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        src_pos_embed=None,
        src_key_padding_mask=None,
    ):

        src_reference_points = self.get_reference_points(src_spatial_shapes, src_valid_ratios, device=src.device)

        output = src
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                src_pos_embed,
                src_reference_points,
                src_spatial_shapes,
                src_level_start_index,
                src_key_padding_mask,
            )
        return output


class DeformTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_checkpoint=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.self_attn = MSDeformAttn(d_model, 1, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.use_checkpoint = use_checkpoint

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward(
        self,
        tgt,
        src,
        tgt_pos_embed,
        tgt_reference_points,
        t2s_reference_points,
        tgt_spatial_shapes,
        src_spatial_shapes,
        tgt_level_start_index,
        src_level_start_index,
        tgt_key_padding_mask,
        src_key_padding_mask,
    ):
        # self attention
        # q = k = self.with_pos_embed(tgt, tgt_pos_embed)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)

        tgt2 = self.self_attn(
            self.with_pos_embed(tgt, tgt_pos_embed),
            tgt_reference_points,
            tgt,
            tgt_spatial_shapes,
            tgt_level_start_index,
            tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_pos_embed),
            t2s_reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt

    def forward(
        self,
        tgt,
        src,
        tgt_pos_embed,
        tgt_reference_points,
        t2s_reference_points,
        tgt_spatial_shapes,
        src_spatial_shapes,
        tgt_level_start_index,
        src_level_start_index,
        tgt_key_padding_mask,
        src_key_padding_mask,
    ):
        if self.use_checkpoint and self.training:
            tgt = checkpoint.checkpoint(
                self._forward,
                tgt,
                src,
                tgt_pos_embed,
                tgt_reference_points,
                t2s_reference_points,
                tgt_spatial_shapes,
                src_spatial_shapes,
                tgt_level_start_index,
                src_level_start_index,
                tgt_key_padding_mask,
                src_key_padding_mask,
            )
        else:
            tgt = self._forward(
                tgt,
                src,
                tgt_pos_embed,
                tgt_reference_points,
                t2s_reference_points,
                tgt_spatial_shapes,
                src_spatial_shapes,
                tgt_level_start_index,
                src_level_start_index,
                tgt_key_padding_mask,
                src_key_padding_mask,
            )
        return tgt


class DeformTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        tgt,
        src,
        tgt_pos_embed,
        t2s_reference_points,
        tgt_spatial_shapes,
        src_spatial_shapes,
        tgt_level_start_index,
        src_level_start_index,
        tgt_valid_ratios,
        src_valid_ratios,
        tgt_key_padding_mask=None,
        src_key_padding_mask=None,
    ):

        tgt_reference_points = self.get_reference_points(tgt_spatial_shapes, tgt_valid_ratios, device=tgt.device)
        t2s_reference_points = t2s_reference_points * src_valid_ratios[:, None]
        # t2s_reference_points = t2s_reference_points[:, :, None] * src_valid_ratios[:, None]

        intermediate = []
        output = tgt
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                src,
                tgt_pos_embed,
                tgt_reference_points,
                t2s_reference_points,
                tgt_spatial_shapes,
                src_spatial_shapes,
                tgt_level_start_index,
                src_level_start_index,
                tgt_key_padding_mask,
                src_key_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
