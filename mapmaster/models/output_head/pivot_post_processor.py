import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mapmaster.models.utils.mask_loss import SegmentationLoss
from mapmaster.utils.misc import nested_tensor_from_tensor_list
from mapmaster.utils.misc import get_world_size, is_available, is_distributed

from .line_matching import pivot_dynamic_matching, seq_matching_dist_parallel


class HungarianMatcher(nn.Module):

    def __init__(self, cost_obj=1., cost_mask=1., coe_endpts=1., cost_pts=2., mask_loss_conf=None):
        super().__init__()
        self.cost_obj, self.cost_mask = cost_obj, cost_mask
        self.coe_endpts = coe_endpts    # end points weight: 1 + coe_endpts
        self.cost_pts = cost_pts        
        self.mask_loss = SegmentationLoss(**mask_loss_conf)

    @torch.no_grad()
    def forward(self, outputs, targets):
        num_decoders, num_classes = len(outputs["ins_masks"]), len(outputs["ins_masks"][0])
        matching_indices = [[[] for _ in range(num_classes)] for _ in range(num_decoders)]
        for dec_id in range(num_decoders):
            for cid in range(num_classes):
                bs, num_queries = outputs["obj_logits"][dec_id][cid].shape[:2]
                
                # 1. obj class cost mat
                dt_probs = outputs["obj_logits"][dec_id][cid].flatten(0, 1).softmax(-1)  # [n_dt, 2], n_dt in a batch
                gt_idxes = torch.cat([tgt["obj_labels"][cid] for tgt in targets])     # [n_gt, ]
                cost_mat_obj = -dt_probs[:, gt_idxes]                                 # [n_dt, n_gt]
                
                # 2. masks cost mat
                dt_masks = outputs["ins_masks"][dec_id][cid].flatten(0, 1)            # [n_dt, h, w]
                gt_masks = torch.cat([tgt["ins_masks"][cid] for tgt in targets])      # [n_gt, h, w]
                cost_mat_mask = 0
                if gt_masks.shape[0] == 0:
                    matching_indices[dec_id][cid] = [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))]
                    continue
                dt_num, gt_num = dt_masks.shape[0], gt_masks.shape[0]
                dt_masks = dt_masks.unsqueeze(1).expand(dt_num, gt_num, *dt_masks.shape[1:]).flatten(0, 1)
                gt_masks = gt_masks.unsqueeze(0).expand(dt_num, gt_num, *gt_masks.shape[1:]).flatten(0, 1)
                
                cost_mat_mask = self.mask_loss(dt_masks, gt_masks, "Matcher").reshape(dt_num, gt_num)
                
                # 3. sequence matching costmat
                dt_pts = outputs["ctr_im"][dec_id][cid].flatten(0, 1)  # [n_dt, n_pts, 2]
                n_pt = dt_pts.shape[1]
                dt_pts = dt_pts.unsqueeze(0).repeat(gt_num, 1, 1, 1).flatten(0, 1)    # [n_gt, n_dt, n_pts, 2] -> [n_gt*n_dt, n_pts, 2]
                gt_pts = targets[0]["points"][cid].to(torch.float32)
                gt_pts = gt_pts.unsqueeze(1).repeat(1, dt_num, 1, 1).flatten(0, 1)  # [n_gt, n_dt, n_pts, 2] -> [n_gt*n_dt, n_pts, 2]

                gt_pts_mask = torch.zeros(gt_num, n_pt, dtype=torch.double, device=gt_pts.device)
                gt_lens = torch.tensor([ll for ll in targets[0]["valid_len"][cid]]) # n_gt
                gt_lens = gt_lens.unsqueeze(-1).repeat(1, dt_num).flatten()
                for i, ll in enumerate(targets[0]["valid_len"][cid]):
                    gt_pts_mask[i][:ll] = 1
                gt_pts_mask = gt_pts_mask.unsqueeze(1).unsqueeze(-1).repeat(1, dt_num, 1, n_pt).flatten(0, 1)   
                cost_mat_seqmatching = torch.cdist(gt_pts, dt_pts, p=1) * gt_pts_mask                # [n_gt*n_dt, n_pts, n_pts]
                cost_mat_seqmatching = seq_matching_dist_parallel(
                    cost_mat_seqmatching.detach().cpu().numpy(), 
                    gt_lens, 
                    self.coe_endpts).reshape(gt_num, dt_num).transpose(1, 0)  #[n_gt, n_dt]
                cost_mat_seqmatching = torch.from_numpy(cost_mat_seqmatching).to(cost_mat_mask.device)
                
                # 4. sum mat
                sizes = [len(tgt["obj_labels"][cid]) for tgt in targets]
                C = self.cost_obj * cost_mat_obj + self.cost_mask * cost_mat_mask + self.cost_pts * cost_mat_seqmatching
                C = C.view(bs, num_queries, -1).cpu()
                indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(C.split(sizes, -1))]
                
                matching_indices[dec_id][cid] = [self.to_tensor(i, j) for i, j in indices]
                
        return matching_indices

    @staticmethod
    def to_tensor(i, j):
        return torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)



class SetCriterion(nn.Module):
    def __init__(self, criterion_conf, matcher, sem_loss_conf=None, no_object_coe=1.0, collinear_pts_coe=1.0, coe_endpts=1.0):  
        super().__init__()
        self.matcher = matcher
        self.criterion_conf = criterion_conf
        self.register_buffer("empty_weight", torch.tensor([1.0, no_object_coe]))
        self.register_buffer("collinear_pt_weight", torch.tensor([collinear_pts_coe, 1.0]))
        self.coe_endpts = coe_endpts
        
        self.sem_loss_conf = sem_loss_conf
        self.mask_loss = SegmentationLoss(**sem_loss_conf["mask_loss_conf"])

    def forward(self, outputs, targets):
        matching_indices = self.matcher(outputs, targets)
        ins_msk_loss, pts_loss, collinear_pts_loss, pt_logits_loss = \
            self.criterion_instance(outputs, targets, matching_indices)
        ins_obj_loss = self.criterion_instance_labels(outputs, targets, matching_indices)
        losses = {"ins_msk_loss": ins_msk_loss, "ins_obj_loss": ins_obj_loss,
                  "pts_loss": pts_loss, "collinear_pts_loss": collinear_pts_loss,
                  "pt_logits_loss": pt_logits_loss}
        if self.sem_loss_conf is not None:
            losses.update({"sem_msk_loss": self.criterion_semantice_masks(outputs, targets)})
        losses = {key: self.criterion_conf['weight_dict'][key] * losses[key] for key in losses}
        return sum(losses.values()), losses

    def criterion_instance(self, outputs, targets, matching_indices):
        loss_masks, loss_pts, loss_collinear_pts, loss_logits = 0, 0, 0, 0
        device = outputs["ins_masks"][0][0].device
        num_decoders, num_classes = len(matching_indices), len(matching_indices[0])
        for i in range(num_decoders):
            w = self.criterion_conf['decoder_weights'][i]
            for j in range(num_classes):
                num_instances = sum(len(t["obj_labels"][j]) for t in targets)
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=device)
                if is_distributed() and is_available():
                    torch.distributed.all_reduce(num_instances)
                num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
                indices = matching_indices[i][j]
                src_idx = self._get_src_permutation_idx(indices)  # dt
                tgt_idx = self._get_tgt_permutation_idx(indices)  # gt

                # instance masks
                src_masks = outputs["ins_masks"][i][j][src_idx]
                tgt_masks = [t["ins_masks"][j] for t in targets]
                tgt_masks, _ = nested_tensor_from_tensor_list(tgt_masks).decompose()
                tgt_masks = tgt_masks.to(src_masks)[tgt_idx]
                loss_masks += w * self.mask_loss(src_masks, tgt_masks, "Loss").sum() / num_instances

                # prepare tgt points 
                src_ctrs = outputs["ctr_im"][i][j][src_idx]  # [num_queries, o, 2]
                tgt_ctrs = []  # [num_queries, o, 2]
                for info in targets:  # B
                    for pts, valid_len in zip(info["points"][j][tgt_idx[1]], info["valid_len"][j][tgt_idx[1]]): # n_gt
                        tgt_ctrs.append(pts[:valid_len])
                
                # pts match, valid pts loss, collinear pts loss
                n_match_q, n_dt_pts = src_ctrs.shape[0], src_ctrs.shape[1]
                logits_gt = torch.zeros((n_match_q, n_dt_pts), dtype=torch.long, device=src_ctrs.device)
                if n_match_q == 0: # avoid unused parameters
                    loss_pts += w * F.l1_loss(src_ctrs, src_ctrs, reduction='sum')
                    loss_logits += w * F.l1_loss(outputs["pts_logits"][i][j][src_idx].flatten(0, 1), outputs["pts_logits"][i][j][src_idx].flatten(0, 1), reduction="sum")
                    continue

                for ii, (src_pts, tgt_pts) in enumerate(zip(src_ctrs, tgt_ctrs)): # B=1, traverse matched query pairs
                    n_gt_pt = len(tgt_pts) 
                    weight_pt = torch.ones((n_gt_pt), device=tgt_pts.device)
                    weight_pt[0] += self.coe_endpts
                    weight_pt[-1] += self.coe_endpts
                    cost_mat = torch.cdist(tgt_pts.to(torch.float32), src_pts, p=1)
                    _, matched_pt_idx = pivot_dynamic_matching(cost_mat.detach().cpu().numpy())
                    matched_pt_idx = torch.tensor(matched_pt_idx)
                    # match pts loss
                    loss_match = w * F.l1_loss(src_pts[matched_pt_idx], tgt_pts, reduction="none").sum(dim=-1)   # [n_gt_pt, 2] -> [n_gt_dt]
                    loss_match = (loss_match * weight_pt).sum() / weight_pt.sum()
                    loss_pts += loss_match / num_instances
                    # interpolate pts loss
                    loss_collinear_pts += w * self.interpolate_loss(src_pts, tgt_pts, matched_pt_idx) / num_instances
                    # pt logits
                    logits_gt[ii][matched_pt_idx] = 1
                loss_logits += w * F.cross_entropy(outputs["pts_logits"][i][j][src_idx].flatten(0, 1), logits_gt.flatten(), self.collinear_pt_weight) / num_instances

        loss_masks /= (num_decoders * num_classes)
        loss_pts /= (num_decoders * num_classes)
        loss_logits /= (num_decoders * num_classes)
        loss_collinear_pts /= (num_decoders * num_classes)

        return loss_masks, loss_pts, loss_collinear_pts, loss_logits
    
    def interpolate_loss(self, src_pts, tgt_pts, matched_pt_idx):
        # 1. pick collinear pt idx
        collinear_idx = torch.ones(src_pts.shape[0], dtype=torch.bool)
        collinear_idx[matched_pt_idx] = 0
        collinear_src_pts = src_pts[collinear_idx]
        # 2. interpolate tgt_pts
        inter_tgt = torch.zeros_like(collinear_src_pts)
        cnt = 0
        for i in range(len(matched_pt_idx)-1):
            start_pt, end_pt = tgt_pts[i], tgt_pts[i+1]
            inter_num = matched_pt_idx[i+1] - matched_pt_idx[i] - 1
            inter_tgt[cnt:cnt+inter_num] = self.interpolate(start_pt, end_pt, inter_num)
            cnt += inter_num
        assert collinear_src_pts.shape[0] == cnt
        # 3. cal loss
        if cnt > 0:
            inter_loss = F.l1_loss(collinear_src_pts, inter_tgt, reduction="sum") / cnt
        else:
            inter_loss = F.l1_loss(collinear_src_pts, inter_tgt, reduction="sum")
        return inter_loss
    
    @staticmethod
    def interpolate(start_pt, end_pt, inter_num):
        res = torch.zeros((inter_num, 2), dtype=start_pt.dtype, device=start_pt.device)
        num_len = inter_num + 1  # segment num.
        for i in range(1, num_len):
            ratio = i / num_len
            res[i-1] = (1 - ratio) * start_pt + ratio * end_pt
        return res

    def criterion_instance_labels(self, outputs, targets, matching_indices):
        loss_labels = 0
        num_decoders, num_classes = len(matching_indices), len(matching_indices[0])
        for i in range(num_decoders):
            w = self.criterion_conf['decoder_weights'][i]
            for j in range(num_classes):
                indices = matching_indices[i][j]
                idx = self._get_src_permutation_idx(indices)  # (batch_id, query_id)
                logits = outputs["obj_logits"][i][j]
                target_classes_o = torch.cat([t["obj_labels"][j][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(logits.shape[:2], 1, dtype=torch.int64, device=logits.device)
                target_classes[idx] = target_classes_o
                loss_labels += (w * F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight))
        loss_labels /= (num_decoders * num_classes)
        return loss_labels

    def criterion_semantice_masks(self, outputs, targets):
        loss_masks = 0
        num_decoders, num_classes = len(outputs["sem_masks"]), len(outputs["sem_masks"][0])
        for i in range(num_decoders):
            w = self.sem_loss_conf['decoder_weights'][i]
            for j in range(num_classes):
                dt_masks = outputs["sem_masks"][i][j]  # (B, 2, H, W)
                gt_masks = torch.stack([t["sem_masks"][j] for t in targets], dim=0)  # (B, H, W)
                loss_masks += w * self.mask_loss(dt_masks[:, 1, :, :], gt_masks).mean()
        loss_masks /= (num_decoders * num_classes)
        return loss_masks

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class PivotMapPostProcessor(nn.Module):
    def __init__(self, criterion_conf, matcher_conf, pivot_conf, map_conf,
                 sem_loss_conf=None, no_object_coe=1.0, collinear_pts_coe=1.0, coe_endpts=0.0):
        super(PivotMapPostProcessor, self).__init__()
        self.criterion = SetCriterion(criterion_conf, HungarianMatcher(**matcher_conf), sem_loss_conf, no_object_coe, collinear_pts_coe, coe_endpts)
        self.ego_size = map_conf['ego_size']
        self.map_size = map_conf['map_size']
        self.line_width = map_conf['line_width']
        self.num_pieces = pivot_conf['max_pieces']  # (10, 2, 30)
        self.num_classes = len(self.num_pieces)
        self.class_indices = torch.tensor(list(range(self.num_classes)), dtype=torch.int).cuda()

    def forward(self, outputs, targets=None):
        if self.training:
            targets = self.refactor_targets(targets)
            return self.criterion.forward(outputs, targets)
        else:
            return self.post_processing(outputs)


    def refactor_targets(self, targets):
        # only support bs == 1
        targets_refactored = []
        targets["masks"] = targets["masks"].cuda()
        for key in [0, 1, 2]: # map type
            targets["points"][key]  = targets["points"][key].cuda()[0]  # [0] remove batch dim
            targets["valid_len"][key] = targets["valid_len"][key].cuda()[0]  # [0] remove batch dim

        for instance_mask in targets["masks"]:  # bs, only support bs == 1
            sem_masks, ins_masks, ins_objects = [], [], []
            for i, mask_pc in enumerate(instance_mask):  # class
                sem_masks.append((mask_pc > 0).float())
                unique_ids = torch.unique(mask_pc, sorted=True)[1:]
                ins_num = unique_ids.shape[0]
                pt_ins_num = len(targets["points"][i])
                if pt_ins_num == ins_num:
                    ins_msk = (mask_pc.unsqueeze(0).repeat(ins_num, 1, 1) == unique_ids.view(-1, 1, 1)).float()
                else:
                    ins_msk = np.zeros((pt_ins_num, *self.map_size), dtype=np.uint8)
                    for j, ins_pts in enumerate(targets["points"][i]):
                        ins_pts_tmp = ins_pts.clone()
                        ins_pts_tmp[:, 0] *= self.map_size[0]
                        ins_pts_tmp[:, 1] *= self.map_size[1]
                        ins_pts_tmp = ins_pts_tmp.cpu().data.numpy().astype(np.int32)
                        cv2.polylines(ins_msk[j], [ins_pts_tmp[:, ::-1]], False, color=1, thickness=self.line_width)
                    ins_msk = torch.from_numpy(ins_msk).float().cuda()
                assert len(ins_msk) == len(targets["points"][i])
                ins_obj = torch.zeros(pt_ins_num, dtype=torch.long, device=unique_ids.device)
                ins_masks.append(ins_msk)
                ins_objects.append(ins_obj)
            targets_refactored.append({
                "sem_masks": sem_masks, 
                "ins_masks": ins_masks, 
                "obj_labels": ins_objects,
                "points": targets["points"],
                "valid_len": targets["valid_len"],
                })
        return targets_refactored

    def post_processing(self, outputs):
        batch_results, batch_masks = [], []
        batch_size = outputs["obj_logits"][-1][0].shape[0]
        for i in range(batch_size):
            points, scores, labels = [None], [-1], [0]
            masks = np.zeros((self.num_classes, *self.map_size)).astype(np.uint8)
            instance_index = 1
            for j in range(self.num_classes):
                pred_scores, pred_labels = torch.max(F.softmax(outputs["obj_logits"][-1][j][i], dim=-1), dim=-1)
                keep_ids = torch.where((pred_labels == 0).int())[0]  # fore-ground
                if keep_ids.shape[0] == 0:
                    continue
                keypts = outputs["ctr_im"][-1][j][i][keep_ids].cpu().data.numpy()  # [P, N, 2]
                keypts[:, :, 0] *= self.map_size[0]
                keypts[:, :, 1] *= self.map_size[1]

                valid_pt_idx = F.softmax(outputs["pts_logits"][-1][j][i][keep_ids], dim=-1)[:,:,1].cpu().data.numpy() > 0.5  # [P, N]
                valid_pt_idx[:, 0] = 1
                valid_pt_idx[:, -1] = 1

                for k, (dt_pts, dt_score) in enumerate(zip(keypts, pred_scores[keep_ids])):
                    select_pt = dt_pts[valid_pt_idx[k]]
                    cv2.polylines(masks[j], [select_pt.astype(np.int32)[:, ::-1]], False, color=instance_index, thickness=1)
                    instance_index += 1
                    points.append(select_pt)
                    scores.append(self._to_np(dt_score).item())
                    labels.append(j + 1)
            batch_results.append({'map': points, 'confidence_level': scores, 'pred_label': labels})
            batch_masks.append(masks)
        return batch_results, batch_masks

    @staticmethod
    def _to_np(tensor):
        return tensor.cpu().data.numpy()

    