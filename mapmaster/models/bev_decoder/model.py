import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.bev_decoder.transformer import Transformer
from mapmaster.models.bev_decoder.deform_transformer import DeformTransformer

class TransformerBEVDecoder(nn.Module):
    def __init__(self, key='im_bkb_features', **kwargs):
        super(TransformerBEVDecoder, self).__init__()
        self.bev_encoder = Transformer(**kwargs)
        self.key = key

    def forward(self, inputs):
        assert self.key in inputs
        feats = inputs[self.key]
        fuse_feats = feats[-1]
        fuse_feats = fuse_feats.reshape(*inputs['images'].shape[:2], *fuse_feats.shape[-3:])
        fuse_feats = torch.cat(torch.unbind(fuse_feats, dim=1), dim=-1)

        cameras_info = {
            'extrinsic': inputs.get('extrinsic', None),
            'intrinsic': inputs.get('intrinsic', None),
            'ida_mats': inputs.get('ida_mats', None),
            'do_flip': inputs['extra_infos'].get('do_flip', None)
        }

        _, _, bev_feats = self.bev_encoder(fuse_feats, cameras_info=cameras_info)

        return {"bev_enc_features": list(bev_feats)}

class DeformTransformerBEVEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(DeformTransformerBEVEncoder, self).__init__()
        self.bev_encoder = DeformTransformer(**kwargs)

    def forward(self, inputs):
        assert "im_bkb_features" in inputs
        feats = inputs["im_bkb_features"]
        for i in range(len(feats)):
            feats[i] = feats[i].reshape(*inputs["images"].shape[:2], *feats[i].shape[-3:])
            feats[i] = feats[i].permute(0, 2, 3, 1, 4)
            feats[i] = feats[i].reshape(*feats[i].shape[:3], -1)
        cameras_info = {
            'extrinsic': inputs.get('extrinsic', None),
            'intrinsic': inputs.get('intrinsic', None),
            'do_flip': inputs['extra_infos'].get('do_flip', None)
        }
        # src_feats: (N, H1 * W1, C)  tgt_feats: # (M, N, H2 * W2, C)
        _, _, bev_feats = self.bev_encoder(feats, cameras_info=cameras_info)
        
        return {
            "bev_enc_features": list(bev_feats),
        }
