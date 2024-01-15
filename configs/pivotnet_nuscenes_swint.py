import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torchvision.transforms import Compose
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from mapmaster.models.network import MapMaster
from mapmaster.engine.core import MapMasterCli
from mapmaster.engine.experiment import BaseExp
from mapmaster.dataset.nuscenes_pivotnet import NuScenesMapDataset
from mapmaster.dataset.transform import Resize, Normalize, ToTensor_Pivot
from mapmaster.utils.misc import get_param_groups, is_distributed


class EXPConfig:
    
    DATA_ROOT = "/data/dataset/public/nuScenes/"
    IMAGE_SHAPE = (900, 1600)

    map_conf = dict(
        dataset_name="nuscenes",
        nusc_root="/data/dataset/public/nuScenes",
        anno_root="/data/dataset/public/nuScenes/customer/pivot-bezier",
        split_dir="assets/splits/nuscenes",
        num_classes=3,
        ego_size=(60, 30),
        map_region=(30, 30, 15, 15),
        map_resolution=0.15,
        map_size=(400, 200),
        mask_key="instance_mask8",
        line_width=8,
        save_thickness=1,
    )

    pivot_conf = dict(
        max_pieces=(10, 2, 30),   # max num of pts in divider / ped / boundary]
    )

    dataset_setup = dict(
        img_key_list=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
        img_norm_cfg=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True),
        input_size=(896, 512),
    )

    model_setup = dict(
        # image-branch
        im_backbone=dict(
            arch_name="swin_transformer",
            bkb_kwargs=dict(arch="tiny", 
                            shift_mode=0, 
                            out_indices=(2, 3), 
                            use_checkpoint=True,
                            pretrained='assets/weights/upernet_swin_tiny_patch4_window7_512x512.pth'),
            ret_layers=2,
            fpn_kwargs=None,
        ),
        bev_decoder=dict(
            arch_name="ipm_deformable_transformer",
            net_kwargs=dict(
                in_channels=[384, 768],
                src_shape=[(32, 336), (16, 168)],
                tgt_shape=(64, 32),
                d_model=256,
                n_heads=8,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                return_intermediate_dec=True,
                dec_n_points=8,
                enc_n_points=8,
                src_pos_encode="learned",
                tgt_pos_encode="learned",
                norm_layer=nn.SyncBatchNorm,
                use_checkpoint=False,
                use_projection=True,
                map_size=map_conf["map_size"],
                map_resolution=map_conf["map_resolution"],
                image_shape=(900, 1600),
                image_order=[2, 1, 0, 5, 4, 3]
            )
        ),
        ins_decoder=dict(
            arch_name="line_aware_decoder",
            net_kwargs=dict(
                decoder_ids=[0, 1, 2, 3, 4, 5],
                in_channels=256,
                num_feature_levels=1,
                mask_classification=True,
                num_classes=1,
                hidden_dim=256,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=6,
                pre_norm=False,
                mask_dim=256,
                enforce_input_project=False,
                query_split=(20, 25, 15),
                max_pieces=pivot_conf["max_pieces"], 
            ),
        ),
        output_head=dict(
            arch_name="pivot_point_predictor",
            net_kwargs=dict(
                in_channel=256,
                num_queries=[20, 25, 15],
                tgt_shape=map_conf['map_size'],
                max_pieces=pivot_conf["max_pieces"],
                bev_channels=256,
                ins_channel=64,
            )
        ),

        post_processor=dict(
            arch_name="pivot_post_processor",
            net_kwargs=dict(
                criterion_conf=dict(
                    weight_dict=dict(
                        sem_msk_loss=3,
                        ins_obj_loss=2, ins_msk_loss=5,
                        pts_loss=30, collinear_pts_loss=10, 
                        pt_logits_loss=2,
                    ),
                    decoder_weights=[0.4, 0.4, 0.4, 0.8, 1.2, 1.6]
                ),
                matcher_conf=dict(
                    cost_obj=2, cost_mask=5,
                    coe_endpts=5,
                    cost_pts=30,
                    mask_loss_conf=dict(
                        ce_weight=1,
                        dice_weight=1,
                    )
                ),
                pivot_conf=pivot_conf,
                map_conf=map_conf,
                sem_loss_conf=dict(
                    decoder_weights=[0.4, 0.8, 1.6, 2.4],
                    mask_loss_conf=dict(ce_weight=1, dice_weight=1)),
                no_object_coe=0.5,
                collinear_pts_coe=0.2,
                coe_endpts=5,
            )
        )
    )

    optimizer_setup = dict(
        base_lr=2e-4,
        wd=1e-4,
        backb_names=["backbone"],
        backb_lr=5e-5,
        extra_names=[],
        extra_lr=5e-5,
        freeze_names=[],
    )

    scheduler_setup = dict(
        milestones=[0.7, 0.9],
        gamma=0.2,
    )

    metric_setup = dict(
        map_resolution=map_conf["map_resolution"],
        iou_thicknesses=(1,),
        cd_thresholds=(0.2, 0.5, 1.0, 1.5, 5.0)
    )
    
    VAL_TXT = [
        "assets/splits/nuscenes/val.txt", 
    ]


class Exp(BaseExp):
    def __init__(self, batch_size_per_device=1, total_devices=8, max_epoch=60, **kwargs):
        super(Exp, self).__init__(batch_size_per_device, total_devices, max_epoch)

        self.exp_config = EXPConfig()
        self.data_loader_workers = 1
        self.print_interval = 10
        self.dump_interval = 1
        self.eval_interval = 1
        self.seed = 0
        self.num_keep_latest_ckpt = 1
        self.ckpt_oss_save_dir = None
        self.enable_tensorboard = True

        milestones = self.exp_config.scheduler_setup["milestones"]
        self.exp_config.scheduler_setup["milestones"] = [int(x * max_epoch) for x in milestones]

        lr_ratio_dict = {32: 2, 16: 1.5, 8: 1, 4: 1, 2: 0.5, 1: 0.5}
        assert total_devices in lr_ratio_dict, "Please set normal devices!"
        for k in ['base_lr', 'backb_lr', 'extra_lr']:
            self.exp_config.optimizer_setup[k] = self.exp_config.optimizer_setup[k] * lr_ratio_dict[total_devices]
        self.evaluation_save_dir = None

    def _configure_model(self):
        model = MapMaster(self.exp_config.model_setup)
        return model

    def _configure_train_dataloader(self):
        from mapmaster.dataset.sampler import InfiniteSampler

        dataset_setup = self.exp_config.dataset_setup

        transform = Compose(
            [
                Resize(img_scale=dataset_setup["input_size"]),
                Normalize(**dataset_setup["img_norm_cfg"]),
                ToTensor_Pivot(),
            ]
        )

        train_set = NuScenesMapDataset(
            img_key_list=dataset_setup["img_key_list"],
            map_conf=self.exp_config.map_conf,
            transforms=transform,
            data_split="training",
        )

        if is_distributed():
            sampler = InfiniteSampler(len(train_set), seed=self.seed if self.seed else 0)
        else:
            sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size_per_device,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=sampler is None,
            drop_last=True,
            sampler=sampler,
        )
        self.train_dataset_size = len(train_set)
        return train_loader

    def _configure_val_dataloader(self):

        dataset_setup = self.exp_config.dataset_setup

        transform = Compose(
            [
                Resize(img_scale=dataset_setup["input_size"]),
                Normalize(**dataset_setup["img_norm_cfg"]),
                ToTensor_Pivot(),
            ]
        )

        val_set = NuScenesMapDataset(
            img_key_list=dataset_setup["img_key_list"],
            map_conf=self.exp_config.map_conf,
            transforms=transform,
            data_split="validation",
        )

        if is_distributed():
            sampler = DistributedSampler(val_set, shuffle=False)
        else:
            sampler = None

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )

        self.val_dataset_size = len(val_set)
        return val_loader

    def _configure_test_dataloader(self):
        pass

    def _configure_optimizer(self):
        optimizer_setup = self.exp_config.optimizer_setup
        optimizer = AdamW(get_param_groups(self.model, optimizer_setup))
        return optimizer

    def _configure_lr_scheduler(self):
        scheduler_setup = self.exp_config.scheduler_setup
        iters_per_epoch = len(self.train_dataloader)
        scheduler = MultiStepLR(
            optimizer=self.optimizer,
            gamma=scheduler_setup["gamma"],
            milestones=[int(v * iters_per_epoch) for v in scheduler_setup["milestones"]],
        )
        return scheduler

    def training_step(self, batch):
        batch["images"] = batch["images"].float().cuda()
        outputs = self.model(batch)
        return self.model.module.post_processor(outputs["outputs"], batch["targets"])

    def test_step(self, batch):
        with torch.no_grad():
            batch["images"] = batch["images"].float().cuda()
            outputs = self.model(batch)
            results, dt_masks = self.model.module.post_processor(outputs["outputs"])
        self.save_results(batch["extra_infos"]["token"], results, dt_masks)

    def save_results(self, tokens, results, dt_masks):
        if self.evaluation_save_dir is None:
            self.evaluation_save_dir = os.path.join(self.output_dir, "evaluation", "results")
            if not os.path.exists(self.evaluation_save_dir):
                os.makedirs(self.evaluation_save_dir, exist_ok=True)
        for (token, dt_res, dt_mask) in zip(tokens, results, dt_masks):
            save_path = os.path.join(self.evaluation_save_dir, f"{token}.npz")
            np.savez_compressed(save_path, dt_mask=dt_mask, dt_res=dt_res)


if __name__ == "__main__":
    MapMasterCli(Exp).run()
