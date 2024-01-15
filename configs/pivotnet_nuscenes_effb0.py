import torch.nn as nn
from mapmaster.engine.core import MapMasterCli
from pivotnet_nuscenes_swint import EXPConfig, Exp

EXPConfig.model_setup["im_backbone"] = dict(
            arch_name="efficient_net",
            ret_layers=2,
            fpn_kwargs=None,
            bkb_kwargs=dict(
                model_name='efficientnet-b0',
                in_channels=3,
                out_stride=32,
                with_head=False,
                with_cp=True,
                norm_layer=nn.SyncBatchNorm,
                weights_path="assets/weights/efficientnet-b0-355c32eb.pth",
            ),
        )

EXPConfig.model_setup['bev_decoder']["net_kwargs"].update(
    dict(
        in_channels=[112, 320],
    )
)

class ExpDev(Exp):
    def __init__(self, batch_size_per_device=1, total_devices=8, max_epoch=60, **kwargs):
        super(ExpDev, self).__init__(batch_size_per_device, total_devices, max_epoch, **kwargs)
        self.exp_config = EXPConfig()

if __name__ == "__main__":
    MapMasterCli(ExpDev).run()
    
