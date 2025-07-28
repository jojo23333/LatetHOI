import torch
import logging
import os
import d2.utils.comm as comm

import numpy as np
from typing import Any
from modeling.backbone.hands import xyz_to_wrist_xyz, wrist_xyz_to_xyz, xyz_relative_xyz, pose_l2_dist, get_hand_pose_condition
from modeling.diffusion.edm_losses import VELoss, VPLoss, EDMLoss
from modeling.build import create_gaussian_diffusion
from utils.visualize import HandMotionVisualizer

from utils.rotation_conversions import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle

DEBUG = False
WRIST_SCALE = 1
JOINT_SCALE = 5

class MANODiffusionWrapper:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("d2")
        ############################################
        # Use Gaussian Diffusion Implementation
        self.model = model.model
        self.gd = create_gaussian_diffusion(cfg)
        self.loss = self.gd.training_losses
        ############################################
        # Use EDM implementation
        # self.model = model
        # self.loss = eval(cfg.DIFFUSION.LOSS)()
        ############################################
        self.device = next(self.model.parameters()).device

        self.batch_transforms = self.build_batch_aug()
        self.coord_type = cfg.DIFFUSION.COORD_TYPE

        if DEBUG:
            self.vis = HandMotionVisualizer(flat_hand_mean=True, interactive=False)
        self.sample_batch = None

    def build_batch_aug(self):
        def normalize_rot_vec_sequence(vec_seq):
            norm_rot_vec = torch.norm(vec_seq, dim=-1, keepdim=True)
            min_norm = torch.min(norm_rot_vec, dim=1, keepdim=True).values
            base_norm = torch.floor((min_norm +torch.pi)/(2*torch.pi)) * 2*torch.pi
            vec_seq = (norm_rot_vec - base_norm)/norm_rot_vec * vec_seq
            return vec_seq

        def normlize_rot(data):
            for k, v in data.items():
                if 'rot' in k or 'pose' in k:
                    b, nf, _ = data[k].shape
                    rot_vec = data[k].view(-1, 3)
                    # NOTE VERSION 1 6D rotation
                    rot_vec_6d = axis_angle_to_rotation_6d(rot_vec)
                    data[k] = (rot_vec_6d.view(b, nf, -1) + 1)/2
                    # NOTE VERSION 2
                    # rot_vec = normalize_rot_vec_sequence(rot_vec)
                    # rot_vec = rot_vec / torch.pi
                    # data[k] = rot_vec.view(b, nf, -1)
                    # assert (data[k] >= -1).all() and (data[k] <= 1).all() 
                elif 'arti' in k:
                    data[k] = data[k] / torch.pi
            return data

        def normalize_trans(data):
            data['trans_l'] = (data['trans_l'] - data['trans_obj'][:, 0:1, :] + 0.8)  / 1.6
            data['trans_r'] = (data['trans_r'] - data['trans_obj'][:, 0:1, :] + 0.8)  / 1.6
            data['trans_obj'] = (data['trans_obj'] - data['trans_obj'][:, 0:1, :] + 0.8) / 1.6
            return data
        
        def normalize_world_coord(data):
            batch_pose = data['world_coord']
            right_hand = batch_pose[:, :, :21, :]
            left_hand = batch_pose[:, :, 21:, :]
            relative_right_hand = right_hand[..., :20, :] - right_hand[:, :, 20:21, :]
            relative_left_hand = left_hand[..., :20, :] - left_hand[:, :, 20:21, :]

            mean_wrist_f0 = data['trans_obj'][:, None, 0:1, :]
            right_wrist = right_hand[:, :, 20:21, :] - mean_wrist_f0
            left_wrist = left_hand[:, :, 20:21, :] - mean_wrist_f0
            
            data['joint_r'] = (relative_right_hand + 0.2) / 0.4
            data['joint_l'] = (relative_left_hand + 0.2) / 0.4
            data['wrist_r'] = (right_wrist + 0.8) / 1.6
            data['wrist_l'] = (left_wrist + 0.8) / 1.6
            return data
            
        def normalize_obj_rel_coord(data):
            batch_pose = data['world_coord']
            vol_pose = data['vol_world_coord']
            data['joint_r'] = batch_pose[:, :, :63]
            data['joint_l'] = batch_pose[:, :, 63:]
            data['vol_r'] = vol_pose[:, :, :63]
            data['vol_l'] = vol_pose[:, :, 63:]
            return data

        transforms = []
        return transforms

    def postprocess(self, data):
    #     for k in data.keys():
    #         if 'rot' in k or 'pose' in k:
    #             # NOTE VERSION 1 6D rotation
    #             b, nf, _ = data[k].shape
    #             rot_vec_6d = data[k].reshape(-1, 6) * 2 - 1
    #             rot_vec = rotation_6d_to_axis_angle(rot_vec_6d)
    #             data[k] = rot_vec.view(b, nf, -1)
    #             # # NOTE VERSION 2
    #             # data[k] = data[k] * torch.pi
    #         elif 'arti' in k:
    #             data[k] = data[k] * torch.pi
    #     if 'trans_obj' in data.keys():
    #         data['trans_obj'] = data['trans_obj'] * 1.6 - 0.8
    #         data['trans_r'] = data['trans_r'] * 1.6 - 0.8 + data['trans_obj'][:, 0:1, :]
    #         data['trans_l'] = data['trans_l'] * 1.6 - 0.8 + data['trans_obj'][:, 0:1, :]
        return data

    def preprocess(self, data):
        data = comm.to_device(data, self.device)
        for transform in self.batch_transforms:
            data = transform(data)

        data, cond = self.model.get_gen_target(data)
        joint_valid = torch.ones((1, 1, 1), device=self.device)
        # for k in data.keys():
        #     print(k, data[k].mean(), data[k].mean(), data[k].max())
        return data, cond, joint_valid

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def __call__(self, data) -> Any:
        batch, cond, valid = self.preprocess(data)
        if self.sample_batch is None:
            self.sample_batch = batch
        if DEBUG:
            for i in range(1):
                d = {k:v[i].cpu().numpy().astype(np.float32) for k,v in self.postprocess(batch).items()}
                d['name'] = data['name'][i]
                d['i'] = data['i'][i]
                if 'left' in d['name']:
                    print(d['i'])
                    self.vis.render_seq(d, f"./exps/debug/runtime{d['i']}_{d['name']}.mp4")
                    import ipdb; ipdb.set_trace()
                    print(d)

        loss_d = self.loss(self.model, batch, condition=cond)
        loss = {k+'_loss': v.mean() for k, v in loss_d.items()}
        # loss_d = self.set_loss(loss)
        # loss_d = self.get_reg_loss(batch, batch_output, weight, loss_d)

        # if loss.isnan().any():
        #     torch.save({
        #         'model': self.model,
        #         'pose_batch': batch,
        #         'data': data,
        #         'cfg': self.cfg,
        #         'losses': loss
        #     }, os.path.join(self.cfg.OUTPUT_DIR, f'cuda{comm.get_rank()}_NAN_DEBUG.pth'))
        return loss

class KeypointDiffusionWrapper:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("d2")
        self.model = model
        self.loss = eval(cfg.DIFFUSION.LOSS)()
        self.device = next(self.model.parameters()).device
        
        if cfg.DIFFUSION.POSE_MODELING == 'wrist_xyz':
            self.pose_transform = xyz_to_wrist_xyz
            self.inv_pose_transform = wrist_xyz_to_xyz
        elif cfg.DIFFUSION.POSE_MODELING == 'xyz':
            self.pose_transform = xyz_relative_xyz
            self.inv_pose_transform = None
        else:
            self.pose_transform = lambda x: x
            self.inv_pose_transform = lambda x: x
            # TODO here
        self.get_condition = lambda x: get_hand_pose_condition(x, cfg.DIFFUSION.CONDITION)
        self.batch_transforms = self.build_batch_aug()
        self.coord_type = cfg.DIFFUSION.COORD_TYPE
        from visualize.hand_visualizer import HandPoseVisualizer
        self.vis = HandPoseVisualizer()
    
    def build_batch_aug(self):
        from datasets.transforms import BatchRandomRotate, BatchRandomNoise
        transforms = []
        rotate_x = self.cfg.DATASET.AUG.ROTATE_X
        rotate_y = self.cfg.DATASET.AUG.ROTATE_Y
        rotate_z = self.cfg.DATASET.AUG.ROTATE_Z
        noise_scale = self.cfg.DATASET.AUG.NOISE
        if rotate_z[0] > 0:
            transforms.append(BatchRandomRotate(axis='z', angle=[rotate_z[1], rotate_z[2]], p=rotate_z[0]))
        if rotate_x[0] > 0:
            transforms.append(BatchRandomRotate(axis='x', angle=[rotate_x[1], rotate_x[2]], p=rotate_x[0]))
        if rotate_y[0] > 0:
            transforms.append(BatchRandomRotate(axis='y', angle=[rotate_y[1], rotate_y[2]], p=rotate_y[0]))
        # transforms.append(BatchRandomNoise(noise_scale=noise_scale))
        return transforms

    def preprocess(self, data):
        data = comm.to_device(data, self.device)
        for transform in self.batch_transforms:
            data = transform(data)

        pose_batch = self.pose_transform(data[self.coord_type].to(self.device))
        condition = self.get_condition(pose_batch)
        if 'joint_valid' in data:
            joint_valid = data['joint_valid']
        else:
            joint_valid = torch.ones(pose_batch.shape[0:3], dtype=torch.bool, device=pose_batch.device)
        return pose_batch, condition, joint_valid

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def __call__(self, data) -> Any:
        pose_batch, condition, valid = self.preprocess(data)
        if DEBUG:
            for i in range(20):
                self.vis.export_simple_vis(self.inv_pose_transform(pose_batch)[i*20].cpu().numpy(), f"./exps/debug/runtime_sanity{i}_{data['name'][i]}.mp4")
        loss = self.loss(net=self.model, batch_data=pose_batch, labels=condition)
        loss = (loss * valid.unsqueeze(-1)).mean()

        if loss.isnan().any():
            torch.save({
                'model': self.model,
                # 'weight': weights,
                'pose_batch': pose_batch,
                # 'diffusion': self.diffusion,
                'data': data,
                'cfg': self.cfg,
                'losses': loss
            }, os.path.join(self.cfg.OUTPUT_DIR, f'cuda{comm.get_rank()}_NAN_DEBUG.pth'))
        return {"loss": loss}

    def test(self, data):
        if self.cfg.TEST.MODE == 'inbetween':
            pose_batch = self.pose_transform(data["joint_world"].to(self.device))
            result = {}
            for rate in [10]:
                mask = torch.zeros_like(pose_batch, dtype=torch.bool)
                mask[:, 0::rate, :, :] = 1
                # use trilinear interpolation to verify rationale
                linear_inp_pose = torch.zeros_like(pose_batch)
                linear_inp_pose[mask] = pose_batch[mask]
                # import ipdb; ipdb.set_trace()
                for ii in range(1, rate):
                    pose_condition = torch.cat([pose_batch[:, 0::rate, :, :], pose_batch[:, 0::rate, :, :][:, -1:, :, :]], dim=1)
                    linear_inp_pose[:, ii::rate, :, :] = (1-ii/rate) * pose_condition[:, :-1, :, :] + (ii/rate) * pose_condition[:, 1:, :, :]

                diff_pose = self.sample(pose_batch, mask)

                l2_dist = ((diff_pose[~mask] - pose_batch[~mask]) ** 2).mean()
                l1_dist = (diff_pose[~mask] - pose_batch[~mask]).abs().mean()
                l2_linear_inp = ((linear_inp_pose[~mask] - pose_batch[~mask]) ** 2).mean()
                l1_linear_inp = (linear_inp_pose[~mask] - pose_batch[~mask]).abs().mean()

                result[f"{rate}x_l2_loss"] = l2_dist.item(),
                result[f"{rate}x_l1_loss"] = l1_dist.item(),
                result[f"{rate}x_linear_l2_loss"] =  l2_linear_inp.item(),
                result[f"{rate}x_linear_l1_loss"] = l1_linear_inp.item()
                result[f"{rate}x"] = diff_pose
                result[f"{rate}x_int"] = linear_inp_pose
            result["gt"] = pose_batch
            return result
        elif self.cfg.TEST.MODE == 'inpaint':
            result = {}
            pose_batch = self.pose_transform(data["joint_world"].to(self.device))
            result["gt"] = self.inv_pose_transform(pose_batch)
            rate = self.cfg.TEST.INPAINT_RATE

            bp = torch.ones(*pose_batch.shape[:-1]+(1,), dtype=torch.bool) * (1/rate)
            mask = torch.bernoulli(bp).to(torch.bool).to(self.device)
            mask[:, :, 0, :] = 1
            mask[:, :, 21, :] = 1
            diff_pose = self.sample(pose_batch, mask)

            mask_repeat = mask.repeat(1, 1, 1, 3)
            l2_unmasked = ((diff_pose[mask_repeat] - pose_batch[mask_repeat]) ** 2).mean().sqrt()
            print("l2_unmasked", l2_unmasked)

            if self.inv_pose_transform is not None:
                diff_pose = self.inv_pose_transform(diff_pose)

            result = pose_l2_dist(diff_pose, result['gt'], result)
            result[f"output"] = diff_pose                
            result[f"mask"] = mask[..., 0]
            return result
        elif self.cfg.TEST.MODE == 'unconditional':
            diff_pose = self.sample(data.to(self.device), None)
            if self.inv_pose_transform is not None:
                diff_pose = self.inv_pose_transform(diff_pose)
            return diff_pose
        else:
            raise NotImplementedError
