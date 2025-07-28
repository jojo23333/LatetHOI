import torch
import itertools
import logging
import math
import os
import pickle
import numpy as np
import d2.utils.comm as comm

from os import path as osp
from typing import Any
from tqdm import tqdm
from modeling.diffusion.edm_diffusion import general_sampler, edm_sampler
from modeling.backbone.hands import pose_l2_dist
from utils.visualize import HandMotionVisualizer
from .eval_physic import PhysicEvaluation
from .fit_joint import FitJoint
logger = logging.getLogger(__name__)


GLOBAL_VISUALIZER = None

class EvalGeneration:
    def __init__(self, cfg, diffusion_model, test_split=None) -> None:
        self.cfg = cfg
        self.diffusion = diffusion_model
        # TODO fix this after fixing diffusion edm or gd
        if 'model' in diffusion_model.model.__dict__:
            self.cond_gen = diffusion_model.model.model.cond_keys
        else:
            self.cond_gen = diffusion_model.model.cond_keys
        self.device = diffusion_model.device
        self.logger = logging.getLogger("d2")
        self.split = cfg.TEST.SPLIT if test_split is None else test_split
        self.max_vis = cfg.TEST.VIS_NUM
        self.vis_seq_len = cfg.DATASET.NUM_FRAMES
        self.output_path = osp.join(self.cfg.OUTPUT_DIR, 'generation')
        if comm.is_main_process():
            global GLOBAL_VISUALIZER
            if GLOBAL_VISUALIZER is None:
                GLOBAL_VISUALIZER = HandMotionVisualizer(flat_hand_mean=True, interactive=False, dataset=cfg.DATASET.NAME)
            self.vis = GLOBAL_VISUALIZER
        self.cond_generation = len(self.cond_gen) > 0
        self.eval_ip = cfg.TEST.EVAL_IP
        self.quantative_metric = PhysicEvaluation(cfg)

        if comm.is_main_process():
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path, exist_ok=True)

    def __call__(self, iter, **kwargs) -> Any:
        # if duplicate key, use whatever in cond dictionary
        if self.split == 'test':
            seq, cond, gt = self.cond_sample(kwargs['test_loader'])
            seq.update(cond)
            return self.evaluate(seq, gt, iter, 'mannual_test', kwargs['test_loader'].dataset.unnormalize_input)
        if self.split == 'val':
            seq, cond, gt = self.cond_sample(kwargs['test_loader'])
            seq.update(cond)
            return self.evaluate(seq, gt, iter, 'eval', kwargs['test_loader'].dataset.unnormalize_input)
        if self.split == 'train':
            seq, cond, gt = self.cond_sample(kwargs['train_loader'])
            seq.update(cond)
            return self.evaluate(seq, gt, iter, 'train', kwargs['train_loader'].dataset.unnormalize_input)

    def evaluate(self, seq, gt, iter, name, unnormalize_f, export_vis=True):
        if 'shape_l' not in seq:
            seq['shape_l'] = gt['shape_l']
            seq['shape_r'] = gt['shape_r']
        if 'type_obj' not in seq:
            seq['type_obj'] = gt['type_obj']
        if 'name' not in seq:
            seq["name"] = gt["name"]

        fit_joint = FitJoint()
        result_list = []
        metrics = {}
        num_samples = len(gt['name'])
        for i in range(num_samples):
            self.logger.info(f"Evaluating {i}/{num_samples}")
            pred_cur = unnormalize_f({k: v[i].detach().cpu().numpy() if k!='name' else v[i] for k, v in seq.items()})
            gt_cur = unnormalize_f({k: v[i].detach().cpu().numpy() if k!='name' else v[i] for k, v in gt.items()})
            fitted_cur = fit_joint(pred_cur)
            cur_metric = self.eval_quantative(pred_cur, gt_cur)
            fitted_metric = self.eval_quantative(fitted_cur, gt_cur)
            result_list.append((gt_cur, pred_cur, fitted_cur))
            print("cur")
            self.logger.info(cur_metric)
            print("fitted")
            self.logger.info(fitted_metric)
            metric = {
                **cur_metric,
                **{f"fitted_{k}" :v for k, v in fitted_metric.items()},
            }
            if i == 0:
                metrics.update(metric)
            else:
                metrics = {k: v + metric[k] for k, v in metrics.items()}
            

        # TODO 
        num_samples = sum(comm.all_gather(num_samples))
        metrics = comm.all_gather(metrics)
        metrics_all = {}
        for k in metrics[0]:
            metrics_all[k] = sum([x[k] for x in metrics])
        metrics = {k : v / num_samples for k, v in metrics_all.items()}
        logger.info("Final Metric")
        logger.info(metrics)

        result_list = [item for sublist in comm.all_gather(result_list) for item in sublist]
        if comm.is_main_process() and export_vis:
            output_path = osp.join(self.output_path, name)
            os.makedirs(output_path, exist_ok=True)
            for gt, pred, pred_fit in result_list:
                out_name = osp.join(output_path, f"{gt['name']}_{gt['i']}".replace('/', '_'))
                self.vis.render_seq({"pred": pred}, out_name + '.mp4')
                self.vis.render_seq({"pred_fit": pred_fit}, out_name + '_fit.mp4')
                torch.save({"gt": gt, "pred": pred, "pred_fit":pred_fit}, out_name + '.pth')
        return metrics

    def eval_quantative(self, seq, gt):
        if not self.eval_ip:
            return {}
        metric_l, metric_r = self.quantative_metric(seq)
        gt_metric_l, gt_metric_r = self.quantative_metric(gt)

        return {
            **{f'pred_l_{k}': v for k, v in metric_l.items()},
            **{f'pred_r_{k}': v for k, v in metric_r.items()},
            **{f'gt_l_{k}': v for k, v in gt_metric_l.items()},
            **{f'gt_r_{k}': v for k, v in gt_metric_r.items()},
        }

    def cond_sample(self, loader):
        # if self.cond_generation > 0:
        for batch in loader:
            seq, cond, _ = self.diffusion.preprocess(batch)
            seq.update(cond)
            if self.cfg.MODEL.BACKBONE == 'ho_transformer_co_ar':
                seq, cond = self.auto_regressive_sample(seq)
            else:
                seq, cond = self.sample(seq)
            break
        return seq, cond, batch
        # else:
        #     diff_place_holder = self.diffusion.sample_batch
        #     for k, v in diff_place_holder.items():
        #         diff_place_holder[k] = v[:self.max_vis]
        #     seq, cond = self.sample(diff_place_holder)
        #     return seq, cond, diff_place_holder


    def sample(self, pose_batch, mask=None):
        self.diffusion.model.eval()
        data, cond = self.diffusion.model.get_gen_target(pose_batch)
        data = comm.to_device(data, self.device)
        cond = comm.to_device(cond, self.device) if cond is not None else None
        if isinstance(pose_batch, dict):
            gt_latent_tensor = self.diffusion.model.dict_2_tensor(data, cond)
        else:
            gt_latent_tensor = data
        # latent_tensor = torch.randn_like(gt_latent_tensor)
        # mask = torch.zeros_like(latent_tensor, dtype=torch.bool)
        # mask[:, :, 444:460] = 1
        diff_pose = self.diffusion.gd.p_sample_loop(
            self.diffusion.model,
            gt_latent_tensor.shape,
            model_kwargs={
                "condition": cond,
                'y':{
                    # "inpainting_mask": mask,
                    # "inpainted_motion": gt_latent_tensor.to(torch.float32),
                }
            }
        )
        # with torch.no_grad():
        #     diff_pose = edm_sampler(
        #         self.diffusion.model,
        #         latent_tensor,
        #         gt_latent=gt_latent_tensor,
        #         condition=cond,
        #         inpaint_mask=mask,
        #         num_steps=1024,
        #         S_churn=40,
        #         S_min=0.05,
        #         S_max=50,
        #         S_noise=1.00
        #     )
        self.diffusion.model.train()
        if isinstance(pose_batch, dict):
            diff_pose = self.diffusion.model.tensor_2_dict(diff_pose, cond, merge_pcd=True)
        else:
            diff_pose = diff_pose
        return diff_pose, cond

    def auto_regressive_sample(self, pose_batch, cond_frames=4):
        self.diffusion.model.eval()

        print(pose_batch.keys())
        num_target_frames = pose_batch['rot_r'].shape[1]
        result = pose_batch
        # import ipdb; ipdb.set_trace()
        for i in tqdm(range(cond_frames, num_target_frames-1)):
            cur_batch = {k: result[k][..., i-cond_frames:i+1, :] for k in result.keys() if k!='arti_obj'}
            cur_batch['arti_obj'] = result['arti_obj'][:, i-cond_frames:i+1]
            data, cond = self.diffusion.model.model.get_gen_target(cur_batch)
            data = comm.to_device(data, self.device)
            cond = comm.to_device(cond, self.device) if cond is not None else None

            latent_tensor = self.diffusion.model.model.dict_2_tensor(data)
            latent_tensor = torch.randn_like(latent_tensor)
            with torch.no_grad():
                diff_pose = edm_sampler(
                    self.diffusion.model,
                    latent_tensor,
                    gt_latent=latent_tensor,
                    condition=cond,
                    inpaint_mask=None,
                    num_steps=256,
                    S_churn=40,
                    S_min=0.05,
                    S_max=50,
                    S_noise=1.003
                )
            diff_pose = self.diffusion.model.model.tensor_2_dict(diff_pose)
            for k in diff_pose.keys():
                result[k][..., i:i+1, :] = diff_pose[k].cpu()

        return result, {}
