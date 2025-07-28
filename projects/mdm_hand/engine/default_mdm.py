import torch
import logging
import weakref
import logging
import os
import json
import imageio
import trimesh
import pyrender
from typing import Any

import numpy as np
import d2.utils.comm as comm
from d2.engine.checkpoints import GeneralCheckpointer
from d2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from d2.engine.defaults import create_ddp_model, DefaultTrainer
from d2.solver.build import get_default_optimizer_params
from d2.solver import build_lr_scheduler
from d2.utils.logger import setup_logger
from d2.utils.events import get_event_storage
from d2.engine import hooks

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datetime import datetime

from modeling.vae.helpers import point2point_signed
from modeling.vae.models import FullBodyGraspNet
from modeling.vae.models_hand_object import HOIGraspNet
from d2.engine.train_loop import HookBase
# from ignite.contrib.metrics import ROC_AUC

from engine.evaluation.eval_mdm import EvalGeneration

logger = logging.getLogger('d2')

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


class CustomTrainingHooks(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self):
        pass

    @property
    def iter(self):
        return self.trainer.iter

    @property
    def end_of_epoch(self):
        return (self.iter+1) % len(self.trainer.train_loader) == 0
    
    @property
    def begin_of_epoch(self):
        return self.iter % len(self.trainer.train_loader) == 0
    
    @property
    def epoch_num(self):
        return self.iter // len(self.trainer.train_loader)

    @property
    def model(self):
        return self.trainer.wrapped_model

    def before_step(self):
        pass

    def after_step(self):
        pass

class DiffTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg, [CustomTrainingHooks()], ModelWrapper=DiffWrapper)
        self.evaluator = EvalGeneration(cfg, self.wrapped_model)

    def build_optimizer(self, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
        )
        return torch.optim.AdamW(
            params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

    def build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_model(self, cfg): 
        from d2.modeling import build_backbone
        backbone = build_backbone(cfg)
        return backbone

    def run_step(self):
        if "_trainer" not in self.__dict__:
            self.train_loader = self.build_dataset_loader(self.cfg, self.optimizer)
            self._trainer = (AMPTrainer if self.cfg.AMP_ENABLED else SimpleTrainer)(
                self.wrapped_model, self.train_loader, self.optimizer, async_write_metrics=True
            )

        self._trainer.iter = self.iter
        self._trainer.run_step()
        
    def evaluate(self, eval_mode=False):
        self.wrapped_model.eval()
        self.logger.info("Begin Evaluating")
        if "test_loader" not in self.__dict__:
            self.test_loader, sampler = self.build_dataset_loader(self.cfg, is_train=False)

        self.wrapped_model.eval()
        result = {}
        with torch.no_grad():
            loss_all = {}
            for data in self.test_loader:
                with torch.no_grad():
                    loss = self.wrapped_model(data)
                    for k, v in loss.items():
                        if k not in loss_all:
                            loss_all[k] = []
                        v = comm.all_gather(v)
                        v = [x.cpu().item() for x in v]
                        v = np.mean(v)
                        loss_all[k].append(v)
            result = {f'val_{k}': np.mean(v) for k, v in loss_all.items()}

        eval_physic = False
        vis = False
        # eval_physic = (self.iter+1)>self.max_iter or eval_mode
        # if self.cfg.DATASET.NAME.startswith('oakink') or self.cfg.DATASET.NAME.startswith('dex'):
        #     eval_physic = False
        #     vis=False
        # else:
        #     vis=True
        if eval_mode:
            result = self.evaluator(self.test_loader, self.iter, eval_physic=eval_physic, vis=vis)
            logger.info(result)
        # result save to json
        with open(os.path.join(self.cfg.OUTPUT_DIR, f"result_{self.iter}.json"), 'w') as f:
            json.dump(result, f, indent=4)
        self.wrapped_model.train()
        return result

    def build_dataset_loader(self, cfg, is_train=True):
        from d2.engine.defaults import build_dataset, worker_init_reset_seed
        num_workers = cfg.DATASET.NUM_WORKERS
        if is_train:
            batch_size = cfg.TRAINING.BATCH_SIZE
            dataset = build_dataset(cfg, "train")
        else:
            batch_size = cfg.TEST.BATCH_SIZE
            dataset = build_dataset(cfg, cfg.TEST.SPLIT)
        if comm.get_world_size() == 1:
            if is_train:
                sampler = torch.utils.data.RandomSampler(dataset) 
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train, drop_last=is_train)
        print(comm.get_world_size(), is_train, sampler is None and is_train)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            worker_init_fn=worker_init_reset_seed,
            persistent_workers=num_workers > 0,
            drop_last=True#is_train
            # shuffle=sampler
        )
        return loader, sampler

def create_gaussian_diffusion(cfg):
    from modeling.diffusion.respace import SpacedDiffusion, space_timesteps
    from modeling.diffusion import gaussian_diffusion as gd

    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(
        schedule_name='cosine', 
        num_diffusion_timesteps=steps, 
        scale_betas=scale_beta
    )
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        cfg=cfg,
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps
    )

class DiffWrapper:
    def __init__(self, model, cfg, device) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.gd = create_gaussian_diffusion(cfg)
        # self.device = next(self.model.parameters()).device

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def __call__(self, data) -> Any:
        # create gaussian and sample from it
        # import ipdb; ipdb.set_trace()
        # dist = torch.distributions.Normal(mean, variance)
        # z = dist.sample()
        x0 = {'feat': data['feat'].to(self.device)}
        y = {'clip': data['text_features'].to(self.device), 'bps': data['bps_object'].to(self.device)}
        loss_d = self.gd.training_losses(self.model, x0, y)
        return loss_d

    def sample(self, data, unormalize_fn=None):
        y = {'clip': data['text_features'].to(self.device), 'bps': data['bps_object'].to(self.device)}     
        f_shape = data['feat'].shape
        # diff_pose = data['feat'].to(self.device)
        diff_pose = self.gd.p_sample_loop(
            self.model,
            f_shape,
            model_kwargs={'y':y}
        )
        # unormalize diffusion output
        if unormalize_fn is not None:
            diff_pose = unormalize_fn(diff_pose)
        
        if diff_pose.shape[-1] == 108:
            diff_pose_decoded = decode_dim108_dex(diff_pose)
        elif diff_pose.shape[-1] == 75:
            diff_pose_decoded = decode_dim75_grab(diff_pose)

        return diff_pose_decoded

from utils.rotation_conversions import rotation_6d_to_axis_angle
def decode_dim75_grab(x):
    assert x.ndim == 3
    bsz, nf, dim = x.shape
    assert dim == 75
    obj_feat, rhand_feat, lhand_feat = x[..., :9], x[..., 9:42], x[..., 42:75]
    obj_rot, obj_trans = rotation_6d_to_axis_angle(obj_feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3), obj_feat[..., 6:]
    rhand_rot, rhand_pose, rhand_trans = rotation_6d_to_axis_angle(rhand_feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3), rhand_feat[..., 6:30], rhand_feat[..., 30:]
    lhand_rot, lhand_pose, lhand_trans = rotation_6d_to_axis_angle(lhand_feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3), lhand_feat[..., 6:30], lhand_feat[..., 30:]
    
    rhand_trans = rhand_trans + obj_trans
    lhand_trans = lhand_trans + obj_trans
    return {
        'rhand': {
            'global_orient': rhand_rot, 'hand_pose': rhand_pose, 'transl': rhand_trans
        },
        'lhand': {
            'global_orient': lhand_rot, 'hand_pose': lhand_pose, 'transl': lhand_trans
        },
        'obj': {
            'global_orient': obj_rot, 
            'transl': obj_trans
        }
    }
    
def decode_dim108_dex(x):
    assert x.ndim == 3
    bsz, nf, dim = x.shape
    assert dim == 108
    obj_feat, rhand_feat = x[..., :9], x[..., 9:]
    obj_rot, obj_trans = rotation_6d_to_axis_angle(obj_feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3), obj_feat[..., 6:]
    rhand_rot = rotation_6d_to_axis_angle(rhand_feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3)
    rhand_pose = rotation_6d_to_axis_angle(rhand_feat[..., 6:96].reshape(-1, 6)).reshape(bsz, -1, 45)
    rhand_trans = rhand_feat[..., 96:]
    rhand_trans = rhand_trans + obj_trans
    return {
        'rhand': {
            'global_orient': rhand_rot, 'hand_pose': rhand_pose, 'transl': rhand_trans
        },
        'obj': {
            'global_orient': obj_rot, 
            'transl': obj_trans
        }
    }