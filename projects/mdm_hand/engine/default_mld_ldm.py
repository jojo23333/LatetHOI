import torch
import logging
import weakref
import logging
import os
import json
import imageio
import trimesh
import pyrender
import copy
import smplx
from bps_torch.bps import bps_torch
from typing import Any
from tqdm import tqdm

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

    # @property
    # def after_backward(self):
    #     # for name, param in self.trainer.wrapped_model.model.named_parameters():
    #     #     if param.grad is None:
    #     #         print(name)
    #     # import ipdb; ipdb.set_trace()
    #     pass
        

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


from modeling.vae.grabpointnet import GrabPointnet
class MLDLdmTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        from mld.models.architectures.mld_vae import MldVae
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]
        vae_model =  MldVae(nfeats=cfg.MODEL.INPUT_DIM).to('cuda')
        logger.info("Loading VAE model")
        vae_model.load_state_dict(torch.load(cfg.DIFFUSION.VAE_CHECKPOINT)["model"])
        for param in vae_model.parameters():
            param.requires_grad = False
        self.vae_model = vae_model
        self.vae_model.eval()
        # self.vae_model = create_ddp_model(vae_model, broadcast_buffers=False)
        super().__init__(cfg, [CustomTrainingHooks()], ModelWrapper=LatentDiffWrapper)

        self.evaluator = EvalGeneration(cfg, self.wrapped_model)
        
        # TODO whether to redo encoding
        # Place holder for potential data augmentation during training\
        self.wrapped_model.vae_model = self.vae_model
        self.wrapped_model.encode = False
        self.wrapped_model.trainer = self

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
        from mld.models.architectures.mld_denoiser import MldDenoiser
        backbone = MldDenoiser(nfeats=cfg.MODEL.INPUT_DIM)
        
        return backbone

    def build_evaluators(self):
        evaluators = []
        return evaluators

    def run_step(self):
        if "_trainer" not in self.__dict__:
            self.train_loader = self.build_dataset_loader(self.cfg, self.optimizer)
            self._trainer = (AMPTrainer if self.cfg.AMP_ENABLED else SimpleTrainer)(
                self.wrapped_model, self.train_loader, self.optimizer, async_write_metrics=True
            )

        self._trainer.iter = self.iter
        self._trainer.run_step()

    def precompute_latent(self, dataset):
        if self.cfg.DATASET.NAME.startswith('grab'):
            return self.precompute_latent_grab(dataset)
        elif self.cfg.DATASET.NAME.startswith('dexycb'):
            return self.precompute_latent_grab(dataset)
        elif self.cfg.DATASET.NAME.startswith('oakink'):
            logger.info("Fake oakink Jumping over latent compute")
        else:
            raise NotImplementedError(f"{self.cfg.DATASET.NAME} not for LDM")
    
    def precompute_latent_grab(self, dataset):
        all_latent = []
        split = dataset.split
        path_to_latent_cache = os.path.join(
            os.path.dirname(self.cfg.DIFFUSION.VAE_CHECKPOINT), 
            f'{self.cfg.DATASET.NAME}_{split}_latent_cache.pt'
        )
        logger.info(f"Precomputing latent in Dir {path_to_latent_cache}")
        with torch.no_grad():
            if os.path.exists(path_to_latent_cache):
                all_latent = torch.load(path_to_latent_cache)
                for i in range(len(dataset)):
                    dataset.update(i, latent=all_latent[i].detach().numpy())
            else:
                for i in tqdm(range(len(dataset))):
                    d = dataset.get_raw(i)
                    x0 = {'feat': torch.from_numpy(d['feat']).to('cuda')}
                    x0['feat'] = x0['feat'].unsqueeze(0)
                    # import ipdb; ipdb.set_trace()
                    z, dist = self.vae_model.encode(x0['feat'], [x0['feat'].shape[1]]*x0['feat'].shape[0])
                    latent = torch.cat(
                        [dist.mean.cpu(), dist.scale.cpu()]
                    , dim=-1)
                    latent = latent.permute(1, 0, 2)
                    # Assuming we are using 1-MLD
                    all_latent.append(latent.cpu().detach())
                    dataset.update(i, latent=latent.cpu().detach().numpy())
                torch.save(all_latent, path_to_latent_cache)
            mean_latent = torch.cat(all_latent, dim=0).mean(dim=0).cpu().detach()
            mean_latent, std_latent = copy.deepcopy(torch.chunk(mean_latent, 2, dim=-1))
            dataset.mean_latent, dataset.std_latent = mean_latent.numpy(), std_latent.numpy()
            logger.info("Precomputing latent done")

    def build_dataset_loader(self, cfg, is_train=True):
        from d2.engine.defaults import build_dataset, worker_init_reset_seed
        num_workers = cfg.DATASET.NUM_WORKERS
        if is_train:
            batch_size = cfg.TRAINING.BATCH_SIZE
            dataset = build_dataset(cfg, "train")
            self.precompute_latent(dataset)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
            dataset = build_dataset(cfg, cfg.TEST.SPLIT)
            self.precompute_latent(dataset)
            # dataset = dataset.clone()
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

    def evaluate(self, eval_mode=False, vis=False):
        self.wrapped_model.eval()
        self.logger.info("Begin Evaluating")
        if "test_loader" not in self.__dict__:
            self.test_loader, sampler = self.build_dataset_loader(self.cfg, is_train=False)

        if not eval_mode:
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
                loss_all = {f'val_{k}': np.mean(v) for k, v in loss_all.items()}
            result = loss_all
        eval_physic = (self.iter+1)>(self.max_iter // 2) or eval_mode
        
        # TODO set every thing to False now, use tools.eval_motion in training eval need to be updated
        vis = False
        eval_physic = False
        if eval_mode:
            result = self.evaluator(self.test_loader, self.iter, eval_physic=eval_physic, vis=vis)
        logger.info(result)
        # result save to json
        with open(os.path.join(self.cfg.OUTPUT_DIR, f"result_{self.iter}.json"), 'w') as f:
            json.dump(result, f, indent=4)
        self.wrapped_model.train()
        return result


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

class LatentDiffWrapper:
    def __init__(self, model, cfg, device) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.gd = create_gaussian_diffusion(cfg)
        BPS_BASE = torch.from_numpy(np.load('./config/bps.npz')['basis'])
        self.bps = bps_torch(custom_basis = BPS_BASE)
        self.dataset = cfg.DATASET.NAME
        if self.dataset.startswith('grab'):
            mano_batch = cfg.TRAINING.BATCH_SIZE * 160
        else:
            mano_batch = cfg.TRAINING.BATCH_SIZE * 96
        self.lhand_model = smplx.create(model_path=cfg.PATH.SMPLX,
                                        model_type='mano',
                                        is_rhand=True,
                                        use_pca=False,
                                        flat_hand_mean=True,
                                        batch_size=mano_batch).to(self.device)
        self.rhand_model = smplx.create(model_path=cfg.PATH.SMPLX,
                                        model_type='mano',
                                        is_rhand=False,
                                        use_pca=False,
                                        flat_hand_mean=True,
                                        batch_size=mano_batch).to(self.device)
        for param in self.lhand_model.parameters():
            param.requires_grad = False
        for param in self.rhand_model.parameters():
            param.requires_grad = False

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def __call__(self, data) -> Any:
        mean, variance = torch.chunk(data['latent'], 2, dim=-1)
        z_ = torch.rand_like(mean).to(mean)
        # Ablation study choices
        z = z_ * variance + mean

        x0 = {
            "feat": z.to(self.device)
        }
        y = {
            'clip': data['text_features'].to(self.device),
            'bps': data['bps_object'].to(self.device),
            # 'vae': self.vae_model,
            'std_feat': data['std_feat'].to(self.device),
            'mean_feat': data['mean_feat'].to(self.device),
        }
        
        loss_d = self.gd.training_losses(self.model, x0, y)
        return loss_d

    def sample(self, data, unormalize_fn=None):
        mean, variance = torch.chunk(data['latent'], 2, dim=-1)
        y = {
            'clip': data['text_features'].to(self.device),
            'bps': data['bps_object'].to(self.device),
            'vae': self.vae_model,
            'std_feat': data['std_feat'].to(self.device),
            'mean_feat': data['mean_feat'].to(self.device),
        }
        f_shape = list(mean.shape)
        # import ipdb; ipdb.set_trace()
        diff_z = self.gd.p_sample_loop(
            self.model,
            f_shape,
            model_kwargs={'y':y}
        )
        
        if self.cfg.DATASET.NAME.startswith('grab'):
            length = [160] * f_shape[0]
            diff_pose_decoded = self.decode_latent_grab(diff_z, length, y)
        elif self.cfg.DATASET.NAME.startswith('dexycb'):
            length = [96] * f_shape[0]
            diff_pose_decoded = self.decode_dim108_dex(diff_z, length, y)
        return diff_pose_decoded

    def decode_latent_grab(self, z, length, y):
        from utils.rotation_conversions import rotation_6d_to_axis_angle
        
        x = self.vae_model.decode(z, length)
        x = x * y.get('std_feat').reshape(1, 1, 75) + y.get('mean_feat').reshape(1, 1, 75)
        # import ipdb; ipdb.set_trace()

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
        
    def decode_dim108_dex(self, z, length, y):
        from utils.rotation_conversions import rotation_6d_to_axis_angle
        x = self.vae_model.decode(z, length)
        x = x * y.get('std_feat').reshape(1, 1, 108) + y.get('mean_feat').reshape(1, 1, 108)

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