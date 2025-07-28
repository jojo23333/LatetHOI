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
class LatentDiffTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]
        vae_model = GrabPointnet(cfg.VAE).to('cuda')
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
        backbone = build_backbone(cfg)
        if comm.is_main_process():
            self.get_model_info(backbone)
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
            return self.precompute_latent_dexycb(dataset)
        elif self.cfg.DATASET.NAME.startswith('oakink'):
            logger.info("Fake oakink Jumping over latent compute")
        else:
            raise NotImplementedError(f"{self.cfg.DATASET.NAME} not for LDM")
    
    def precompute_latent_grab(self, dataset):
        def collate_fn(batch, device):
            batch = copy.deepcopy(batch)
            for k in batch:
                if isinstance(batch[k], np.ndarray):
                    batch[k] = torch.from_numpy(batch[k]).to(device)
            return batch
        all_latent = []
        split = dataset.split
        self.rhand_offset = torch.tensor([0.0957, 0.0064, 0.0062]).to(self.cfg.DEVICE).view(1, 3)
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
                    ldata = d['lhand'] 
                    # trans_rhand is already relative
                    # ldata['trans_rhand'] = (ldata['trans_rhand'] - ldata['trans_obj']).astype(np.float32)
                    ldata = collate_fn(ldata, self.cfg.DEVICE)
                    ldata['trans_rhand'] = ldata['trans_rhand'] + self.rhand_offset
                    rdata = d['rhand']
                    # rdata['trans_rhand'] = (rdata['trans_rhand'] - rdata['trans_obj']).astype(np.float32)
                    rdata = collate_fn(rdata, self.cfg.DEVICE)
                    rdata['trans_rhand'] = rdata['trans_rhand'] + self.rhand_offset
                    lhand_z = self.vae_model(encode=True, **ldata)
                    rhand_z = self.vae_model(encode=True, **rdata)
                    latent = torch.cat(
                        [lhand_z.mean.cpu(), rhand_z.mean.cpu(),
                        lhand_z.scale.cpu(), rhand_z.scale.cpu()]
                    , dim=-1)
                    all_latent.append(latent.cpu().detach())
                    dataset.update(i, latent=latent.cpu().detach().numpy())
                torch.save(all_latent, path_to_latent_cache)
            mean_latent = torch.cat(all_latent, dim=0).mean(dim=0).cpu().detach()
            mean_latent, std_latent = copy.deepcopy(torch.chunk(mean_latent, 2, dim=-1))
            dataset.mean_latent, dataset.std_latent = mean_latent.numpy(), std_latent.numpy()
            logger.info("Precomputing latent done")
    
    def precompute_latent_dexycb(self, dataset):
        def collate_fn(batch, device):
            batch = copy.deepcopy(batch)
            for k in batch:
                if isinstance(batch[k], np.ndarray):
                    batch[k] = torch.from_numpy(batch[k]).to(device)
            return batch
        all_latent = []
        split = dataset.split
        self.rhand_offset = torch.tensor([0.0957, 0.0064, 0.0062]).to(self.cfg.DEVICE).view(1, 3)
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
                    rdata = d['rhand']
                    # rdata['trans_rhand'] = (rdata['trans_rhand'] - rdata['trans_obj']).astype(np.float32)
                    rdata = collate_fn(rdata, self.cfg.DEVICE)
                    rdata['trans_rhand'] = rdata['trans_rhand'] + self.rhand_offset
                    rhand_z = self.vae_model(encode=True, **rdata)
                    latent = torch.cat(
                        [rhand_z.mean.cpu(), rhand_z.scale.cpu()]
                    , dim=-1)
                    all_latent.append(latent.cpu().detach())
                    dataset.update(i, latent=latent.cpu().detach().numpy())
                torch.save(all_latent, path_to_latent_cache)
            mean_latent = torch.cat(all_latent, dim=0).mean(dim=0)
            
            mean_latent, std_latent = copy.deepcopy(torch.chunk(mean_latent, 2, dim=-1))
            dataset.mean_latent, dataset.std_latent = mean_latent.numpy(), std_latent.numpy()
            # dataset.mean_latent, dataset.std_latent = copy.deepcopy(torch.chunk(mean_latent, 2, dim=-1))
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

    def evaluate(self, eval_mode=False):
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
        if self.cfg.DIFFUSION.LDM_SHARE_REPARAM: # this is the setting in paper
            z_ = z_[:, :1, :]
            z = z_ * variance + mean
        elif self.cfg.DIFFUSION.LDM_USE_MEAN:
            z = mean
        elif self.cfg.DIFFUSION.LDM_USE_MEAN_VAR:
            z = data['latent']
        else:
            z = z_ * variance + mean

        x0 = {
            "z": z.to(self.device),
            "feat": data['feat'].to(self.device)
        }
        # VAE seems a bit broken for now, skip it since diffusion works ok
        if self.dataset.startswith('grab'):
            y = {
                'clip': data['text_features'].to(self.device),
                'bps': data['bps_object'].to(self.device),
                "obj_verts": data['obj_verts'].to(self.device),
                # 'vae': self.vae_model,
                "dec_fn": self.decode_latent_grab,
                'std_feat': data['std_feat'].to(self.device),
                'mean_feat': data['mean_feat'].to(self.device),
                'lhand': (data['lhand'], self.lhand_model),
                'rhand': (data['rhand'], self.rhand_model),
            }
        else:
            y = {
                'clip': data['text_features'].to(self.device),
                'bps': data['bps_object'].to(self.device),
                "obj_verts": data['obj_verts'].to(self.device),
                # 'vae': self.vae_model,
                "dec_fn": self.decode_latent_grab,
                'std_feat': data['std_feat'].to(self.device),
                'mean_feat': data['mean_feat'].to(self.device),
                'rhand': (data['rhand'], self.rhand_model),
            }
        # if self.trainer.iter <= self.trainer.max_iter * 4 // 5:
        #     y.pop('vae')
        
        loss_d = self.gd.training_losses(self.model, x0, y)
        return loss_d

    def sample(self, data, unormalize_fn=None):
        y = {
            'clip': data['text_features'].to(self.device),
            'bps': data['bps_object'].to(self.device),
            "obj_verts": data['obj_verts'].to(self.device),
            'vae': self.vae_model,
            "dec_fn": self.decode_latent_grab,
            'std_feat': data['std_feat'].to(self.device),
            'mean_feat': data['mean_feat'].to(self.device),
        }
        f_shape = list(data['feat'].shape)
        f_shape[-1] = f_shape[-1] + data['latent'].shape[-1] // 2
        diff_pose = self.gd.p_sample_loop(
            self.model,
            f_shape,
            model_kwargs={'y':y}
        )
        # unormalize diffusion out put
        ##########################################################
        # Used for debug
        # mean, variance = torch.chunk(data['latent'], 2, dim=-1)
        # z_ = torch.rand_like(mean).to(mean)
        # if self.cfg.DIFFUSION.LDM_SHARE_REPARAM:
        #     z_ = z_[:, :1, :]
        # z = mean
        # x0 = {
        #     "z": z.to(self.device),
        #     "feat": data['feat'].to(self.device)
        # }
        
        # diff_pose_decoded = self.decode_latent_dexycb(x0['z'], x0['feat'], y)
        # # diff_pose_decoded = self.decode_latent_grab(x0['z'], x0['feat'], y)
        ##########################################################
        if self.dataset.startswith('grab') or self.dataset.startswith('oakink'):
            # Do ablation on latent here
            # Instead of using latent by diffusion model, resample from gaussian
            ################################################################################################
            zgen = torch.from_numpy(np.random.normal(0., 1., size=(diff_pose.shape[0], diff_pose.shape[1], 32))).to(diff_pose)
            # diff_pose_decoded = self.decode_latent_grab(zgen, diff_pose[..., 32:], y)
            diff_pose_decoded = self.decode_latent_grab(zgen, data['feat'].to(self.device), y)
            ################################################################################################
            # diff_pose_decoded = self.decode_latent_grab(diff_pose[..., :32], diff_pose[..., 32:], y)
        else:
            diff_pose_decoded = self.decode_latent_dexycb(diff_pose[..., :16], diff_pose[..., 16:], y)
        return diff_pose_decoded

    def decode_latent_grab(self, z, feat, y):
        from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
        bsz, nf, _ = z.shape
        z_l, z_r = z[:, :, :16], z[:, :, 16:32]
        feat = feat * y['std_feat'].view(bsz, 1, -1) + y['mean_feat'].view(bsz, 1, -1)
        # import ipdb; ipdb.set_trace()
        obj_rot_mat, obj_trans = rotation_6d_to_matrix(feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3, 3), feat[..., 6:9]    
        trans_l, trans_r = feat[..., 9:12], feat[..., 12:]
        
        obj_rot_mat = obj_rot_mat.to(torch.float32)
        obj_rot = matrix_to_axis_angle(obj_rot_mat).to(torch.float32)
        # 1. Transfer trans_l & trans_r to relative trans
        # 2. Mirror trans_l 
        # rel_trans_l, rel_trans_r = trans_l - obj_trans, trans_r - obj_trans
        rel_trans_l , rel_trans_r = trans_l, trans_r
        rel_trans_l = rel_trans_l * torch.tensor([-1, 1, 1]).to(rel_trans_l).view(1, 1, 3)
        HAND_OFFSET = torch.tensor([0.0957, 0.0064, 0.0062]).to(self.cfg.DEVICE).view(1, 1, 3)
        rel_trans_l, rel_trans_r = rel_trans_l + HAND_OFFSET, rel_trans_r + HAND_OFFSET
        
        # 3.  get vertices Apply rotation
        # 4.  calculate bps and mirrored bps
        verts = torch.matmul(y['obj_verts'].view(bsz, 1, -1, 3).to(torch.float32), obj_rot_mat)
        verts_mirrored = verts * torch.tensor([-1, 1, 1]).to(verts).view(1, 1, 1, 3)

        lhand_param = self.vae_model(
            verts_object=verts_mirrored.reshape(bsz*nf, -1, 3),
            trans_rhand=rel_trans_l.reshape(bsz*nf, -1),
            global_orient_rhand_rotmat=None,
            fpose_rhand_rotmat=None,
            z_input=z_l.reshape(bsz*nf, -1),
            decode=True
        )
        rhand_param = self.vae_model(
            verts_object=verts.reshape(bsz*nf, -1, 3),
            trans_rhand=rel_trans_r.reshape(bsz*nf, -1),
            global_orient_rhand_rotmat=None,
            fpose_rhand_rotmat=None,
            z_input=z_r.reshape(bsz*nf, -1),
            decode=True
        )
        lhand_param['transl'] = lhand_param['transl'] - HAND_OFFSET.view(1, 3)
        rhand_param['transl'] = rhand_param['transl'] - HAND_OFFSET.view(1, 3)

        # 5. mirror back left hand
        lhand_param['transl'] = lhand_param['transl'] * torch.tensor([-1, 1, 1]).view(1, 3).to(lhand_param['transl'])
        lhand_param['global_orient'] = lhand_param['global_orient'] * torch.tensor([1, -1, -1]).view(1, 3).to(lhand_param['transl'])
        lhand_param['hand_pose'] = lhand_param['hand_pose'].reshape(-1, 15, 3) * torch.tensor([1, -1, -1]).view(1, 1, 3).to(lhand_param['transl'])
        # lhand_param.pop('fullpose')
        # rhand_param.pop('fullpose')
        
        # 6. reshape things back to right shape
        lhand_param['transl'] = lhand_param['transl'].reshape(bsz, nf, 3)
        lhand_param['global_orient'] = lhand_param['global_orient'].reshape(bsz, nf, 3)
        lhand_param['hand_pose'] = lhand_param['hand_pose'].reshape(bsz, nf, 45)
        rhand_param['transl'] = rhand_param['transl'].reshape(bsz, nf, 3)
        rhand_param['global_orient'] = rhand_param['global_orient'].reshape(bsz, nf, 3)
        rhand_param['hand_pose'] = rhand_param['hand_pose'].reshape(bsz, nf, 45)

        # 6. Add back translation relative to object
        lhand_param['transl'] = lhand_param['transl'] + obj_trans
        rhand_param['transl'] = rhand_param['transl'] + obj_trans
        
        return {
            'rhand': rhand_param,
            'lhand': lhand_param,
            "obj": {
                'global_orient': obj_rot, 
                'transl': obj_trans
            },
            'z': z,
        }
    
    def decode_latent_dexycb(self, z, feat, y):
        from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
        bsz, nf, _ = z.shape
        z_r = z
        feat = feat * y['std_feat'].view(bsz, 1, -1) + y['mean_feat'].view(bsz, 1, -1)
        # import ipdb; ipdb.set_trace()
        obj_rot_mat, obj_trans = rotation_6d_to_matrix(feat[..., :6].reshape(-1, 6)).reshape(bsz, -1, 3, 3), feat[..., 6:9]    
        trans_r = feat[..., 9:]
        
        obj_rot_mat = obj_rot_mat.to(torch.float32)
        obj_rot = matrix_to_axis_angle(obj_rot_mat).to(torch.float32)
        # 1. Transfer trans_l & trans_r to relative trans
        # 2. Mirror trans_l 
        # rel_trans_l, rel_trans_r = trans_l - obj_trans, trans_r - obj_trans
        rel_trans_r =  trans_r
        HAND_OFFSET = torch.tensor([0.0957, 0.0064, 0.0062]).to(self.cfg.DEVICE).view(1, 1, 3)
        rel_trans_r = rel_trans_r + HAND_OFFSET
        
        # 3.  get vertices Apply rotation
        # 4.  calculate bps and mirrored bps
        verts = torch.matmul(y['obj_verts'].view(bsz, 1, -1, 3).to(torch.float32), obj_rot_mat)

        rhand_param = self.vae_model(
            verts_object=verts.reshape(bsz*nf, -1, 3),
            trans_rhand=rel_trans_r.reshape(bsz*nf, -1),
            global_orient_rhand_rotmat=None,
            fpose_rhand_rotmat=None,
            z_input=z_r.reshape(bsz*nf, -1),
            decode=True
        )
        rhand_param['transl'] = rhand_param['transl'] - HAND_OFFSET.view(1, 3)

        # 6. reshape things back to right shape
        rhand_param['transl'] = rhand_param['transl'].reshape(bsz, nf, 3)
        rhand_param['global_orient'] = rhand_param['global_orient'].reshape(bsz, nf, 3)
        rhand_param['hand_pose'] = rhand_param['hand_pose'].reshape(bsz, nf, 45)

        # 6. Add back translation relative to object
        rhand_param['transl'] = rhand_param['transl'] + obj_trans
        
        return {
            'rhand': rhand_param,
            "obj": {
                'global_orient': obj_rot, 
                'transl': obj_trans
            }
        }