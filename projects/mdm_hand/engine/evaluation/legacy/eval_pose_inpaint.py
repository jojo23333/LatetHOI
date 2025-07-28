import torch
import itertools
import logging
import math
import numpy as np
import d2.utils.comm as comm

from os import path as osp
from typing import Any
from modeling.diffusion.edm_diffusion import general_sampler
from modeling.backbone.hands import pose_l2_dist
from utils.visualize import HandPoseVisualizer
logger = logging.getLogger(__name__)

class InpaintEvaluator:
    def __init__(self, cfg, diffusion_model) -> None:
        self.cfg = cfg
        self.diffusion = diffusion_model
        self.device = diffusion_model.device
        self.logger = logging.getLogger("d2")
        self.visualizer = HandPoseVisualizer()
        self.max_vis = 3
        
        rate = cfg.TEST.INPAINT_RATE
        bp = torch.ones((200, 512, 20, 42, 1), dtype=torch.bool) * (1/rate)
        mask = torch.bernoulli(bp).to(torch.bool).to(self.device)
        mask = mask.view(200, 512, 20, 1, 42, 1).repeat(1, 1, 5, 1, 1, 1)
        self.mask = mask.view(200, 512, 100, 42, 1)
        
    def __call__(self, iter, data_loader, vis=False) -> Any:
        results = []
        for ii, data in enumerate(data_loader):
            self.logger.info(f"{ii}/{len(data_loader)}")
            result = self.eval(data, ii)
            results.append(comm.to_device(result, 'cpu'))
            break
        comm.synchronize()
        results = comm.all_gather(results)
        results = list(itertools.chain(*results))
        self.logger.info("Test Result:")
        metrics = {}        
        if comm.is_main_process():
            for k in results[0].keys():
                if 'loss' in k or 'dist' in k:
                    mean_value = np.mean([x[k] for x in results])
                    metrics[k] = mean_value
                    self.logger.info(f"Metric {k}: {mean_value}")
            torch.save(results, osp.join(self.cfg.OUTPUT_DIR, f'val_result_{iter}.pth'))
            if vis:
                self.vis(results, iter)
        return metrics

    def vis(self, results, iter):
        try:
            preds = [(result['output'], result['gt'], result['mask']) for result in results]
            for i, (pred, gt, mask) in enumerate(preds):
                pred = pred[0].cpu().numpy()
                gt = gt[0].cpu().numpy()
                mask = mask[0].cpu().numpy()
                self.visualizer.export_vis(
                    [pred],
                    osp.join(self.cfg.OUTPUT_DIR, f'vis/iter{iter}/inpaint_', f'compare_{i}.mp4'),
                    gt_sequence=gt,
                    mask=mask,
                    fps=self.cfg.DATASET.FPS,
                    title=['pred', 'GT']
                )
                if i > self.max_vis:
                    break
        except Exception as e:
            print(e)
            pass


    def sample(self, pose_batch, mask):
        self.diffusion.model.eval()
        with torch.no_grad():
            solver = 'heun' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'euler'
            discretization = 'edm' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'iddpm'
            schedule = 'linear' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'vp'
            scaling = 'none' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'vp'
            diff_pose = general_sampler(
                self.diffusion.model,
                torch.randn_like(pose_batch),
                gt_latent=pose_batch,
                inpaint_mask=mask,
                num_steps=256,
                solver=solver,
                discretization=discretization,
                schedule=schedule,
                scaling=scaling,
                S_churn=40,
                S_min=0.05,
                S_max=50,
                S_noise=1.003,
                condition=self.diffusion.get_condition(pose_batch)
            )
        self.diffusion.model.train()
        return diff_pose

    def eval(self, data, ii):
        result = {}
        pose_batch = self.diffusion.pose_transform(data["joint_world"].to(self.device))
        result["gt"] = self.diffusion.inv_pose_transform(pose_batch)

        mask = self.mask[ii][:pose_batch.shape[0], :pose_batch.shape[1], :pose_batch.shape[2], :1]
        mask[:, :, 20, :] = 1
        mask[:, :, 41, :] = 1
        diff_pose = self.sample(pose_batch, mask)

        # mask_repeat = mask.repeat(1, 1, 1, 3)
        # l2_unmasked = ((diff_pose[mask_repeat] - pose_batch[mask_repeat]) ** 2).mean().sqrt()
        # print("l2_unmasked", l2_unmasked)

        if self.diffusion.inv_pose_transform is not None:
            diff_pose = self.diffusion.inv_pose_transform(diff_pose)

        result = pose_l2_dist(diff_pose, result['gt'], result, prefix='Inpaint')
        result[f"output"] = diff_pose                
        result[f"mask"] = mask[..., 0]
        return result


from datasets.a101.utils.transforms import Camera, cam2world_assemblyhands, world2cam_assemblyhands
from torch.utils.data import DataLoader

class PoseRefineEvaluator(InpaintEvaluator):
    def __init__(self, cfg, diffusion_model) -> None:
        self.cfg = cfg
        self.diffusion = diffusion_model
        self.device = diffusion_model.device
        self.logger = logging.getLogger("d2")
        self.visualizer = HandPoseVisualizer()
        self.max_vis = 3
        self.masked_portion = 0.6
        self.random_mask = False
        bp = torch.ones((100, 1024, 100, 42, 1), dtype=torch.bool) * self.masked_portion
        self.mask = torch.bernoulli(bp).to(torch.bool).to(self.device)

        dataset = DetectorOuput()
        if comm.get_world_size() == 1:
            sampler = None
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=0,
            pin_memory=True,
            sampler=sampler
        )
        self.coord_type = self.cfg.DIFFUSION.COORD_TYPE

    def __call__(self, iter, _, vis=True) -> Any:
        return super().__call__(iter, self.data_loader, vis)

    def vis(self, results, iter):
        print("Start Visualize")
        try:
            preds = [(result['diff_pose'], result['input_pose'], result['gt_pose'], result['mask'], result['joint_valid']) for result in results]
            for i, (diff_pose, input_pose, gt_pose, mask, joint_valid) in enumerate(preds):
                diff_pose = diff_pose[0].cpu().numpy()
                input_pose = input_pose[0].cpu().numpy()
                gt_pose = gt_pose[0].cpu().numpy()
                mask = mask[0].cpu().numpy()
                joint_valid = joint_valid[0].cpu().numpy()
                self.visualizer.export_vis(
                    [diff_pose, input_pose],
                    osp.join(self.cfg.OUTPUT_DIR, f'vis/iter{iter}/pose_refine', f'compare_{i}.mp4'),
                    gt_sequence=gt_pose,
                    mask=mask,
                    fps=self.cfg.DATASET.FPS,
                    title=['output', 'input'],
                    gt_column=False
                )
                if i > self.max_vis:
                    break
        except Exception as e:
            print(e)
            pass

    def get_mask(self, score, joint_valid, ii):
        bsz, nf, nj = score.shape
        mask = torch.ones((bsz, nf, nj, 1), device=self.device, dtype=torch.bool)

        if self.random_mask:
            mask = self.mask[ii, :bsz, :nf, :nj, :]
        else:
            # get mask target by score
            score[~joint_valid] = 1
            _, indice = torch.sort(score.view(bsz, -1))
            indice_2d = torch.stack([torch.floor(indice / nj), (indice % nj)], dim=-1).to(torch.int64)

            mask_cnt = (nf*nj - torch.sum(~joint_valid.view(bsz, -1), dim=-1)) * self.masked_portion
            mask_cnt = mask_cnt.to(torch.int)
            for i in range(bsz):
                mask_idx = indice_2d[i, :mask_cnt[i], :]
                mask[i, mask_idx[:, 0], mask_idx[:, 1], :] = False
        mask[~joint_valid] = False
        return mask

    def sample(self, pose_batch, mask, jump_step=0):
        self.diffusion.model.eval()
        with torch.no_grad():
            solver = 'heun' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'euler'
            discretization = 'edm' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'iddpm'
            schedule = 'linear' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'vp'
            scaling = 'none' if self.cfg.DIFFUSION.TYPE == 'EDM' else 'vp'
            diff_pose = general_sampler(
                self.diffusion.model,
                torch.randn_like(pose_batch),
                gt_latent=pose_batch,
                inpaint_mask=mask,
                num_steps=256,
                solver=solver,
                discretization=discretization,
                schedule=schedule,
                scaling=scaling,
                S_churn=40,
                S_min=0.05,
                S_max=50,
                S_noise=1.003,
                jump_step=jump_step,
                condition=self.diffusion.get_condition(pose_batch)
            )
        self.diffusion.model.train()
        return diff_pose

    def eval(self, data, ii):
        # NOTE no normalize to wrist yet
        input_pose = self.diffusion.pose_transform(data[self.coord_type].to(self.device))
        gt_pose = self.diffusion.pose_transform(data[f"gt_{self.coord_type}"].to(self.device))
        score = data["score"]
        joint_valid = data["joint_valid"]

        if self.diffusion.inv_pose_transform is not None:
            input_pose_ = self.diffusion.inv_pose_transform(input_pose)
            gt_pose_ = self.diffusion.inv_pose_transform(gt_pose)

        result = pose_l2_dist(input_pose_, gt_pose_, {}, valid=joint_valid, prefix="input")
        # result = pose_l2_dist(input_pose_, gt_pose_, result, valid=~joint_valid, prefix=f"input_invalid")

        for jump_step in [160, 168, 176, 184, 192]:
            # mask = self.get_mask(score, joint_valid, ii)
            mask = torch.zeros(input_pose.shape[:3]+(1,), device=self.device, dtype=torch.bool)
            mask[:, :, 20, :] = True
            mask[:, :, 41, :] = True
            input_pose[:, :, 20, :] = gt_pose[:, :, 20, :]
            input_pose[:, :, 41, :] = gt_pose[:, :, 41, :]
            diff_pose_ = self.sample(input_pose, mask, jump_step)
            if self.diffusion.inv_pose_transform is not None:
                input_pose_ = self.diffusion.inv_pose_transform(input_pose)
                diff_pose_ = self.diffusion.inv_pose_transform(diff_pose_)

            result = pose_l2_dist(diff_pose_, gt_pose_, result, valid=joint_valid, prefix=f"j{jump_step}_output")
            # result = pose_l2_dist(diff_pose_, gt_pose_, result, valid=~joint_valid, prefix=f"j{jump_step}_output_invalid")
        
        diff_pose = diff_pose_
        gt_pose = gt_pose_
        input_pose = input_pose_
        # if self.diffusion.inv_pose_transform is not None:
        #     input_pose = self.diffusion.inv_pose_transform(input_pose)
        #     diff_pose = self.diffusion.inv_pose_transform(diff_pose)
        #     gt_pose = self.diffusion.inv_pose_transform(gt_pose)

        result = pose_l2_dist(diff_pose, gt_pose, result, valid=joint_valid, prefix=f"output")
        result = pose_l2_dist(diff_pose, gt_pose, result, valid=~joint_valid, prefix=f"output_invalid")
        for k, v in result.items():
            if math.isnan(v) and 'invalid' in k:
                result[k] = 0
        result["diff_pose"] = diff_pose
        result["input_pose"] = input_pose
        result["gt_pose"] = gt_pose
        result["mask"] = mask[..., 0]
        result["joint_valid"] = joint_valid
        return result


class DetectorOuput(torch.utils.data.Dataset):
    def __init__(self, path='/mnt/nimble/nimble-dgx/users/muchenli/DATASET/EvoNetOutput.pth', num_frames=60, fps=30):
        import sys
        sys.path.append('/mnt/nimble/nimble-dgx/users/muchenli/assemblyhands-toolkit')
        super().__init__()
        data = torch.load(path, map_location='cpu')
        self.data = {}
        gap = fps // 30
        for i, d in enumerate(data):
            video_name = osp.dirname(d['img_path'])
            img_id = int(osp.basename(d['img_path']).split('.')[0])
            if video_name not in self.data:
                self.data[video_name] = {}
            if img_id not in self.data[video_name]:
                self.data[video_name][img_id] = d
            else: # merge left right hand
                assert (self.data[video_name][img_id]['gt_joint_coord_cam'] == d['gt_joint_coord_cam']).all()
                cur_valid = d['joint_valid']
                prev_valid = self.data[video_name][img_id]['joint_valid']
                self.data[video_name][img_id]['pred_joint_coord_cam'][cur_valid, :] = d['pred_joint_coord_cam'][cur_valid, :]
                self.data[video_name][img_id]['score'][cur_valid] = d['score'][cur_valid]
                self.data[video_name][img_id]['joint_valid'] += cur_valid

        self.valid_sequence = []
        invalid_cnt = 0
        for k in self.data.keys():
            sorted_frames = sorted(self.data[k].items(), key=lambda x: x[0])
            for id_, (img_id, frame) in enumerate(sorted_frames):
                if id_>0 and img_id == sorted_frames[id_-1][0]:
                    import ipdb; ipdb.set_trace()
                start = id_
                end = start + gap*(num_frames-1)
                if end >= len(sorted_frames):
                    break
                end_frame = sorted_frames[end]
                # print(end_frame[0], img_id, end_frame[0] - img_id)
                if end_frame[0] - img_id == 2 * gap * (num_frames-1):
                    self.valid_sequence.append((k, start, end+1, gap))
                else:
                    invalid_cnt = invalid_cnt + 1
            self.data[k] = sorted_frames
        logger.info(f"{len(self.valid_sequence)} valid sequence out of {len(self.valid_sequence) + invalid_cnt} frames")

    def __len__(self):
        return len(self.valid_sequence)

    def __getitem__(self, idx):
        video_name, start, end, gap = self.valid_sequence[idx]
        frames = self.data[video_name][start:end:gap]

        coord_world = []
        coord_cam = []
        gt_coord_cam = []
        gt_coord_world = []
        score = []
        joint_valid = []
        for img_id, x in frames:
            K, R, t = x['cam_params']
            coord_cam.append(torch.from_numpy(x['pred_joint_coord_cam']))
            coord_world.append(torch.from_numpy(cam2world_assemblyhands(x['pred_joint_coord_cam'], R, t)))
            gt_coord_cam.append(torch.from_numpy(world2cam_assemblyhands(x['gt_coord_world'], R, t)))
            gt_coord_world.append(torch.from_numpy(x['gt_coord_world']))
            score.append(torch.from_numpy(x['score']))
            joint_valid.append(torch.from_numpy(x['joint_valid']))
        coord_cam = torch.stack(coord_cam, dim=0)
        coord_world = torch.stack(coord_world, dim=0)
        gt_coord_cam = torch.stack(gt_coord_cam, dim=0)
        gt_coord_world = torch.stack(gt_coord_world, dim=0)
        score = torch.stack(score, dim=0)
        joint_valid = torch.stack(joint_valid, dim=0)

        return {
            "idx": self.valid_sequence[idx],
            "joint_world": coord_world,
            "gt_joint_world": gt_coord_world,
            "joint_cam": coord_cam,
            "gt_joint_cam": gt_coord_cam,
            "score": score,
            "joint_valid": joint_valid,
        }

    def cal_metrics(self, results, gt_name='gt_joint_cam'):
        result = {}
        for batch_result in results:
            bsz, num_frames = batch_result['output'].shape[:2]
            for bid in range(bsz):
                video_name, start, end, gap = batch_result['idx'][bid]
                fids = list(range(start, end, gap))
                for i in range(num_frames):
                    fid = f'{video_name}_fid_{fids[i]}'
                    if fid in result:
                        result[fid].append(batch_result['output_joint'][bid])
                    else:
                        result[fid] = [batch_result['output_joint'][bid]]
        l2_dist = 0
        l2_ldist = 0
        l2_rdist = 0
        for k, v in result.items():
            video_name, fid = k.split('_fid_')
            fid = int(fid)
            output_joint = torch.cat(v, dim=0).mean(dim=0)
            gt_joint = self.data[video_name][fid]
            l2_dist += ((output_joint - gt_joint) ** 2).mean().sqrt()
            l2_ldist += ((output_joint[:20]-gt_joint[:20]) ** 2).mean().sqrt()
            l2_rdist += ((output_joint[21:41]-gt_joint[21:41]) ** 2).mean().sqrt()
        
        print(l2_dist, l2_ldist, l2_rdist)
                    

if __name__ == '__main__':
    from IPython import get_ipython
    from ipdb.__main__ import _get_debugger_cls
    import ipdb, os
    debugger = _get_debugger_cls()
    shell = get_ipython()
    shell.debugger_history_file = os.path.join('./', '.pdbhistory')
    logging.basicConfig(level=logging.INFO)
    d = DetectorOuput('../DATASET/EvoNetOutput.pth')
    a = d[0]