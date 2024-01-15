import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, basic_type='linear'):
        super().__init__()
        self.basic_type = basic_type
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(self.basic_layer(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    def basic_layer(self, n, k):
        if self.basic_type == 'linear':
            return nn.Linear(n, k)
        elif self.basic_type == 'conv':
            return nn.Conv2d(n, k, kernel_size=1, stride=1)
        else:
            raise NotImplementedError

class PivotMapOutputHead(nn.Module):
    def __init__(self, in_channel, num_queries, tgt_shape, max_pieces, bev_channels=-1, ins_channel=64):
        super(PivotMapOutputHead, self).__init__()
        self.num_queries = num_queries
        self.num_classes = len(num_queries)
        self.tgt_shape = tgt_shape
        self.bev_channels = bev_channels  
        self.semantic_heads = None
        if self.bev_channels > 0:
            self.semantic_heads = nn.ModuleList(
                nn.Sequential(nn.Conv2d(bev_channels, 2, kernel_size=1, stride=1)) for _ in range(self.num_classes)
            )

        self.max_pieces = max_pieces  # [10, 2, 30]
        self.pts_split = [num_queries[i]*max_pieces[i] for i in range(len(num_queries))]
        _N = self.num_classes
        _C = ins_channel  
        self.im_ctr_heads = nn.ModuleList(FFN(in_channel, 256, 2 * _C, 3) for _ in range(_N))
        self.pts_cls_heads = nn.ModuleList(FFN((_C)*2, _C*2, 2, 3) for i in range(_N))  
        self.gap_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.coords = self.compute_locations(device='cuda')  # (1, 2, h, w)
        self.coords_head = FFN(2, 256, _C, 3, 'conv')

    def forward(self, inputs):
        num_decoders = len(inputs["mask_features"])
        dt_obj_logit = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        dt_ins_masks = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        im_ctr_coord = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        dt_pivots_logits = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        coords_feats = self.coords_head.forward(self.coords.repeat((inputs["mask_features"][0].shape[0], 1, 1, 1)))

        for i in range(num_decoders):
            x_ins_cw = inputs["mask_features"][i].split(self.num_queries, dim=1)
            x_obj_cw = inputs["obj_scores"][i].split(self.num_queries, dim=1)
            x_qry_cw = inputs["decoder_outputs"][i].split(self.pts_split, dim=1)  # [(b, 200, c), (b, 50, c), (b, 450, c)]
            batch_size = x_qry_cw[0].shape[0]
            for j in range(self.num_classes):
                dt_ins_masks[i][j] = self.up_sample(x_ins_cw[j])  # (B, P, H, W)
                dt_obj_logit[i][j] = x_obj_cw[j]                  # (B, P, 2)
                # im
                num_qry, n_pts = self.num_queries[j], self.max_pieces[j]
                im_feats = self.im_ctr_heads[j](x_qry_cw[j])                             # (bs, n_q * n_pts, 2*c)
                im_feats_tmp = im_feats.reshape(batch_size, num_qry*n_pts*2, -1)             # (bs, n_q*n_pts*2, c)
                im_coords_map = torch.einsum("bqc,bchw->bqhw", im_feats_tmp, coords_feats)  # [bs, n_q*n_pts*2, h, w]                
                im_coords = self.gap_layer(im_coords_map)                               # [bs, n_q * n_pts]
                im_coords = im_coords.reshape(batch_size, num_qry, self.max_pieces[j], 2).sigmoid()
                im_ctr_coord[i][j] = im_coords

                pt_feats = im_feats.reshape(batch_size, num_qry, self.max_pieces[j], -1).flatten(1, 2)       # [bs, n_q * n_pts, 2*C]
                pt_logits = self.pts_cls_heads[j](pt_feats)
                dt_pivots_logits[i][j] = pt_logits.reshape(batch_size, num_qry, self.max_pieces[j], 2)
                
        ret = {"outputs": {"obj_logits": dt_obj_logit, "ins_masks": dt_ins_masks,
                           "ctr_im": im_ctr_coord, "pts_logits": dt_pivots_logits}}
                           
        if self.semantic_heads is not None:
            num_decoders = len(inputs["bev_enc_features"])
            dt_sem_masks = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
            for i in range(num_decoders):
                x_sem = inputs["bev_enc_features"][i]
                for j in range(self.num_classes):
                    dt_sem_masks[i][j] = self.up_sample(self.semantic_heads[j](x_sem))  # (B, P, 2, H, W)
            ret["outputs"].update({"sem_masks": dt_sem_masks})
        return ret

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)

    def compute_locations(self, stride=1, device='cpu'):

        fh, fw = self.tgt_shape

        shifts_x = torch.arange(0, fw * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, fh * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, fh, fw)
        locations[:, 0, :, :] /= fw
        locations[:, 1, :, :] /= fh

        return locations
