import torch.nn as nn
from mapmaster.engine.core import MapMasterCli
from pivotnet_nuscenes_swint import EXPConfig, Exp


EXPConfig.model_setup["im_backbone"] = dict(
            arch_name="resnet",
            ret_layers=2,
            fpn_kwargs=None,
            bkb_kwargs=dict(
                depth=50,
                num_stages=4,
                out_indices=(2, 3),
                frozen_stages=-1,  # do not freeze any layers
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict(
                    type='Pretrained', 
                    checkpoint='assets/weights/resnet50-0676ba61.pth'),  # from pytorch
                with_cp=True,
            ),   
        )

EXPConfig.model_setup['bev_decoder']["net_kwargs"].update(
    dict(
        in_channels=[1024, 2048],
    )
)

class ExpDev(Exp):
    def __init__(self, batch_size_per_device=1, total_devices=8, max_epoch=60, **kwargs):
        super(ExpDev, self).__init__(batch_size_per_device, total_devices, max_epoch, **kwargs)
        self.exp_config = EXPConfig()

if __name__ == "__main__":
    MapMasterCli(ExpDev).run()
    
