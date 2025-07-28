import torch
import logging
import weakref
import logging
import os
import json
import trimesh
import pyrender
import smplx
from typing import Any

import numpy as np
import d2.utils.comm as comm
from d2.engine.defaults import create_ddp_model, DefaultTrainer
from d2.solver import build_lr_scheduler, WarmupCosineLR
from d2.solver.build import get_default_optimizer_params
from d2.utils.events import get_event_storage
# from d2.engine import hooks
from tqdm import tqdm

from torch.nn import functional as F
# from utils.metrics import ObjectContactMetrics

# from modeling.vae.helpers import point2point_signed
# from modeling.vae.models import FullBodyGraspNet
# from modeling.vae.models_hand_object import HOIGraspNet
from d2.engine.train_loop import HookBase
# from ignite.contrib.metrics import ROC_AUC

# from modeling.vae.grabnet import CoarseNet
from modeling.vae.grabpointnet import GrabPointnet
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
)
import trimesh


MAX_DIS = 1000
DEBUG = False
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
    #     for name, param in self.trainer.wrapped_model.model.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     import ipdb; ipdb.set_trace()
    # #     pass
        

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
        # if self.model.cfg.kl_annealing:
        #     self.model.cfg.kl_coef = min(0.5*(self.epoch_num+1) / self.model.cfg.kl_annealing_epoch, 0.5)
        # if self.begin_of_epoch:
        #     self.trainer.storage.put_scalar("KL Coeficient", self.model.cfg.kl_coef)
        #     logger.info("KL Coeficient: {}".format(self.model.cfg.kl_coef))

    def after_step(self):
        pass
        # if self.end_of_epoch:
        #     train_roc_auc_object = self.trainer.wrapped_model.ROC_AUC_object.compute()
        #     train_roc_auc_markers = self.trainer.wrapped_model.ROC_AUC_marker.compute()
        #     self.trainer.storage.put_scalars(**{
        #         "train_roc_auc_object": np.mean(comm.gather(train_roc_auc_object)),
        #         "train_roc_auc_markers": np.mean(comm.gather(train_roc_auc_markers)),
        #     }, smoothing_hint=False)
        #     self.trainer.wrapped_model.ROC_AUC_object.reset()
        #     self.trainer.wrapped_model.ROC_AUC_marker.reset()


class VaeTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg, [CustomTrainingHooks()], VaeWrapper)

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

    def build_dataset_loader(self, cfg, is_train=True):
        from d2.engine.defaults import build_dataset, worker_init_reset_seed
        from torch.utils.data import DataLoader
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
            drop_last=True
            # shuffle=sampler
        )
        return loader, sampler

    def build_model(self, cfg):
        model = GrabPointnet(cfg.VAE)
        # model = HOIGraspNet(cfg.VAE)
        # model = FullBodyGraspNet(cfg.VAE)
        model.to(cfg.DEVICE)
        if cfg.VAE.freeze_pointnet:
            for param in model.pointnet.parameters():
                param.requires_grad = False
        if comm.is_main_process():
            self.get_model_info(model)
        return model

    def evaluate(self, vis=False, write_metric=True):
        self.logger.info("Begin Evaluating")
        if "test_loader" not in self.__dict__:
            self.test_loader, _ = self.build_dataset_loader(self.cfg, is_train=False)

        writable_metrics = self.wrapped_model.evaluate(self.test_loader)
        self.logger.info(str(writable_metrics))
        return writable_metrics

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()


class VaeWrapper:
    def __init__(self, model, cfg, device):
        self.model = model
        # self.ROC_AUC_object = ROC_AUC()
        # self.ROC_AUC_marker = ROC_AUC()
        self.LossL1 = torch.nn.L1Loss(reduction='none')
        self.LossL2 = torch.nn.MSELoss(reduction='none')
        self.device = device
        self.max_vis_num = cfg.TEST.VIS_NUM
        self.cfg = cfg.VAE
        self.cfg_all = cfg
        
        MODEL_DIR = './data/body_models'            
        self.rhand = smplx.create(
            model_path=MODEL_DIR,
            model_type='mano',
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=cfg.TRAINING.BATCH_SIZE,
        ).to(self.device)
        self.rhand_test = smplx.create(
            model_path=MODEL_DIR,
            model_type='mano',
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=cfg.TEST.BATCH_SIZE,
        ).to(self.device)
        self.rh_f = self.rhand.faces
        self.rhand_offset = torch.tensor([[0.0957, 0.0064, 0.0062]]).to(self.device)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def preprocess_input(self, dorig, test=False):
        """
            Put some preprocessing on gpu
        """
        dorig = {k: v if isinstance(v, list) else v.to(self.device) for k, v in dorig.items()}
        if self.cfg_all.DATASET.NAME != 'graspxl':
            return dorig
        from smplx.lbs import batch_rodrigues
        rot_aug = True
        trans_rhand = dorig['trans_rhand'] - dorig['trans_obj']
        rhand_params = {'global_orient': dorig['full_pose'][:, :3],
                        'hand_pose': dorig['full_pose'][:, 3:],
                        'transl': trans_rhand}
        if rhand_params['transl'].shape[0] == self.cfg_all.TEST.BATCH_SIZE:
            hand_model = self.rhand_test
        else:
            hand_model = self.rhand
        with torch.no_grad():
            dorig['verts_rhand'] = hand_model(**rhand_params).vertices.detach()
        rot_mat_obj = batch_rodrigues(dorig['rot_obj']).view(-1, 3, 3).transpose(1, 2).to(torch.float32)
        verts_obj = torch.bmm(dorig['verts_obj'], rot_mat_obj)
        
        global_orient_rhand_rotmat = batch_rodrigues(dorig['full_pose'][:, :3]).view(-1, 3, 3).to(torch.float32)
        fpose_rhand_rotmat = batch_rodrigues(dorig['full_pose'][:, 3:].reshape(-1, 3)).view(-1, 15, 3, 3).to(torch.float32)
        
        data_out = {
            'trans_rhand': trans_rhand,
            'global_orient_rhand_rotmat': global_orient_rhand_rotmat,
            'fpose_rhand_rotmat': fpose_rhand_rotmat,
            'verts_object': verts_obj,
            'verts_rhand': dorig['verts_rhand'],
            'has_contact': dorig['has_contact'],
        }
        if rot_aug and not test:
            # orient = torch.FloatTensor(1, 3).uniform_(-torch.pi/4, torch.pi/4)
            orient = torch.FloatTensor(1, 3).uniform_(-torch.pi, torch.pi)
            orient[:, :-1] = 0
            rot_mats = batch_rodrigues(orient.view(-1, 3)).view(1, 3, 3).to(torch.float32).to(self.device)
            data_out['verts_object'] = torch.matmul(data_out['verts_object'], rot_mats).to(torch.float32)
            data_out['verts_rhand'] = torch.matmul(data_out['verts_rhand'], rot_mats).to(torch.float32)
            data_out['trans_rhand'] = torch.matmul(data_out['trans_rhand'].view(-1, 3)+self.rhand_offset, rot_mats).to(torch.float32).squeeze() - self.rhand_offset
            data_out['global_orient_rhand_rotmat'] = torch.matmul(rot_mats.transpose(1, 2), data_out['global_orient_rhand_rotmat'].view(-1, 3, 3)).to(torch.float32)

            # TODO add this for sronger augmentation that hopefully will bring better performance for dexycb
            orient = torch.FloatTensor(1, 3).uniform_(-torch.pi/4, torch.pi/4)
            orient[:, 1:] = 0
            rot_mats = batch_rodrigues(orient.view(-1, 3)).view(1, 3, 3).to(torch.float32).to(self.device)
            data_out['verts_object'] = torch.matmul(data_out['verts_object'], rot_mats).to(torch.float32)
            data_out['verts_rhand'] = torch.matmul(data_out['verts_rhand'], rot_mats).to(torch.float32)
            data_out['trans_rhand'] = torch.matmul(data_out['trans_rhand'].view(-1, 3)+self.rhand_offset, rot_mats).to(torch.float32).squeeze() - self.rhand_offset
            data_out['global_orient_rhand_rotmat'] = torch.matmul(rot_mats.transpose(1, 2), data_out['global_orient_rhand_rotmat'].view(-1, 3, 3)).to(torch.float32)

        # verts_object, trans_rhand, global_orient_rhand_rotmat, fpose_rhand_rotmat, has_contact
        return data_out

    @property
    def training(self):
        return self.model.training

    def __call__(self, dorig):
        dorig = self.preprocess_input(dorig)
        dorig['trans_rhand'] = dorig['trans_rhand'] + self.rhand_offset
        if self.cfg.freeze_pointnet:
            self.model.pointnet.eval()
        drec_net = self.model(**dorig)
        dorig['trans_rhand'] = dorig['trans_rhand'] - self.rhand_offset
        drec_net['transl'] = drec_net['transl'] - self.rhand_offset
        cur_loss_dict_net, out = self.loss_net(dorig, drec_net)
        return cur_loss_dict_net

    def visualize(self, object, hand, joint=None):
        """
        Render the object and hand together using PyTorch3D.

        Parameters:
        - object (torch.Tensor): Tensor of shape (N, 3) representing object vertices.
        - hand (torch.Tensor): Tensor of shape (M, 3) representing hand vertices.
        - joint (torch.Tensor, optional): Tensor of shape (J, 3) representing joint positions.

        Returns:
        - color (np.ndarray): Rendered image as a NumPy array of shape (H, W, 3).
        """
        # Select the first GPU from CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        first_gpu = cuda_visible.split(",")[0]
        device = torch.device(f"cuda:{0}") if torch.cuda.is_available() else torch.device("cpu")
        
        # Move data to CPU and convert to NumPy
        object_np = object.detach().cpu().numpy()
        hand_np = hand.detach().cpu().numpy()
        
        # Prepare object mesh (point cloud represented as small spheres or dummy faces)
        # Here, we'll represent object as a mesh with no faces by assigning dummy faces
        # Alternatively, you can create a more sophisticated representation
        # For simplicity, we'll assign the same face indices as the hand but offset
        # Note: This is a workaround; for proper point cloud rendering, consider other methods
        num_object_vertices = object_np.shape[0]
        object_faces = np.array([[i, (i+1) % num_object_vertices, (i+2) % num_object_vertices] for i in range(num_object_vertices)])
        
        # Convert object to PyTorch tensors
        object_vertices = torch.from_numpy(object_np).float().to(device)
        object_faces = torch.from_numpy(object_faces.astype(np.int32)).long().to(device)
        
        # Assign a distinct color to object vertices, e.g., blue
        object_colors = torch.ones_like(object_vertices) * torch.tensor([0.1, 0.1, 0.9]).to(device)
        object_colors = object_colors.unsqueeze(0)  # Shape: (1, N, 3)
        object_textures = TexturesVertex(verts_features=object_colors)
        
        object_mesh = Meshes(
            verts=[object_vertices],
            faces=[object_faces],
            textures=object_textures
        )
        
        # Prepare hand mesh
        hand_np = hand.detach().cpu().numpy()
        hand_faces = self.rh_f  # Assuming self.rh_f is a NumPy array of shape (K, 3)
        hand_vertices = torch.from_numpy(hand_np).float().to(device)
        hand_faces = torch.from_numpy(hand_faces.astype(np.int32)).long().to(device)
        
        # Assign a distinct color to hand vertices, e.g., green
        hand_colors = torch.ones_like(hand_vertices) * torch.tensor([0.1, 0.9, 0.1]).to(device)
        hand_colors = hand_colors.unsqueeze(0)  # Shape: (1, M, 3)
        hand_textures = TexturesVertex(verts_features=hand_colors)
        
        hand_mesh = Meshes(
            verts=[hand_vertices],
            faces=[hand_faces],
            textures=hand_textures
        )
        
        # Combine object and hand into a single Meshes object
        combined_mesh = object_mesh.extend(1)  # Extend to have at least one mesh
        combined_mesh = join_meshes_as_scene([combined_mesh, hand_mesh])
        
        # Optionally add joints as a separate point cloud
        if joint is not None:
            joint_np = joint.detach().cpu().numpy()
            joint_vertices = torch.from_numpy(joint_np).float().to(device)
            joint_faces = torch.tensor([], dtype=torch.long).to(device)  # No faces for joints
            
            # Assign a distinct color to joints, e.g., red
            joint_colors = torch.ones_like(joint_vertices) * torch.tensor([1.0, 0.0, 0.0]).to(device)
            joint_colors = joint_colors.unsqueeze(0)  # Shape: (1, J, 3)
            joint_textures = TexturesVertex(verts_features=joint_colors)
            
            # Create dummy faces for joints (each joint is a single vertex, so no faces)
            # PyTorch3D requires at least one face, so we'll skip adding faces for joints
            # Alternatively, represent joints as small spheres or another mesh
            # Here, we'll skip rendering joints due to complexity
            pass  # Joints can be visualized separately if needed
        
        # Initialize camera parameters
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=180)  # Adjust as needed
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        # Define the settings for rasterization and shading
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # Initialize lights
        lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
        
        # Create a Phong renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
        
        # Render the image
        images = renderer(combined_mesh)
        
        # Convert to numpy and return the image
        image = images[0, ..., :3].cpu().numpy()
        
        return image

    def evaluate(self, dataloader):
        # object_contact_metrics = ObjectContactMetrics(self.device, False)
        self.model.eval()

        eval_loss_dict_net = {}
        cnt_vis = 0
        penatration_volume = []
        penatration_depth = []
        for dorig in tqdm(dataloader):
            with torch.no_grad():
                dorig = self.preprocess_input(dorig, test=True)
                dorig['trans_rhand'] = dorig['trans_rhand'] + self.rhand_offset
                drec_net = self.model(**dorig)
                dorig['trans_rhand'] = dorig['trans_rhand'] - self.rhand_offset
                drec_net['transl'] = drec_net['transl'] - self.rhand_offset
                loss_net, out = self.loss_net(dorig, drec_net)
                eval_loss_dict_net = {k: eval_loss_dict_net.get(k, 0.0) + v.item() for k, v in loss_net.items()}
                recon_hand_vertices = out['verts_rhand']
            
            # if comm.is_main_process() and cnt_vis < self.max_vis_num:
                
            #     with torch.no_grad():
            #     # sample and solve for dist_matrix
            #         dorig['trans_rhand'] = dorig['trans_rhand'] + self.rhand_offset
            #         drec = self.model(sample=True, **dorig)
            #         dorig['trans_rhand'] = dorig['trans_rhand'] - self.rhand_offset
            #         drec['transl'] = drec['transl'] - self.rhand_offset
            #         sample_hand_vertices = self.rhand_test(**drec).vertices
            
            #     verts_object_world = dorig['verts_object']
            #     gt_hand_vertices = dorig['verts_rhand']
            #     gt_vis = self.visualize(verts_object_world[0], gt_hand_vertices[0])
            #     reconstruct_vis = self.visualize(verts_object_world[0], recon_hand_vertices[0])
            #     sample_vis = self.visualize(verts_object_world[0], sample_hand_vertices[0])

            #     cat_image = np.concatenate([gt_vis, reconstruct_vis, sample_vis], axis=1)
            #     try:
            #         storage = get_event_storage()
            #         storage.put_image(f'vis_{cnt_vis}', cat_image)
            #         cnt_vis = cnt_vis + 1
            #     except AssertionError:
            #         # import ipdb; ipdb.set_trace()
            #         from PIL import Image
            #         im = Image.fromarray(cat_image)
            #         im.save(os.path.join(self.cfg_all.OUTPUT_DIR, f'vis_{cnt_vis}.jpg'))
            #         cnt_vis = cnt_vis + 0.5
            cnt_vis = cnt_vis + 1
            if cnt_vis > 200:
                break
        # if comm.is_main_process():
        #     storage.put_scalar('eval/penatration_volume', np.mean(penatration_volume))
        #     storage.put_scalar('eval/inter_depth', np.mean(penatration_depth))

        eval_loss_dict_net = {f'eval/{k}': sum(comm.all_gather(v)) / (len(dataloader)*comm.get_world_size()) for k, v in eval_loss_dict_net.items()}
        self.model.train()
        return eval_loss_dict_net


    def loss_net(self, dorig, drec):
        if self.cfg.NAME == 'grabnet':
            device = dorig['verts_object'].device
            dtype = dorig['verts_object'].dtype
            q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
            p_z = torch.distributions.normal.Normal(
                loc=torch.zeros([drec['mean'].shape[0], drec['mean'].shape[1]], requires_grad=False).to(device).type(dtype),
                scale=torch.ones([drec['mean'].shape[0], drec['mean'].shape[1]], requires_grad=False).to(device).type(dtype))
            loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
            
            if drec['transl'].shape[0] == self.cfg_all.TEST.BATCH_SIZE:
                hand_model = self.rhand_test
            else:
                hand_model = self.rhand
            out_put = hand_model(**drec)
            verts_rhand = out_put.vertices
            loss_mesh = self.LossL2(verts_rhand, dorig['verts_rhand']).mean()
            gt_rot_mat = torch.cat([dorig['global_orient_rhand_rotmat'].unsqueeze(1), dorig['fpose_rhand_rotmat']], dim=1).view(-1, 16, 3, 3)
            rot_mat_loss = self.LossL2(gt_rot_mat, drec['fullpose'])
            orient_loss = rot_mat_loss[:, 0, :, :].mean()
            handpose_loss = rot_mat_loss[:, 1:, :, :].mean()

            loss_dict = {
                'loss_kl': loss_kl,
                'loss_mesh': loss_mesh*5000,
                'orient_loss': orient_loss*100,
                'handpose_loss': handpose_loss*100,
            }
            out = {
                "verts_rhand": verts_rhand
            }
            return loss_dict, out
            
        elif self.cfg.NAME == 'hoinet':
            device = dorig['verts_object'].device
            dtype = dorig['verts_object'].dtype

            ################################ loss weight
            if self.cfg.kl_annealing:
                weight_kl = self.cfg.kl_coef
                weight_rec = 0.5 * 30
            else:
                weight_kl = self.cfg.kl_coef
                weight_rec = (1 - self.cfg.kl_coef) * 30

            ################################# KL loss
            q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
            p_z = torch.distributions.normal.Normal(
                loc=torch.zeros([drec['mean'].shape[0], drec['mean'].shape[1]], requires_grad=False).to(device).type(dtype),
                scale=torch.ones([drec['mean'].shape[0], drec['mean'].shape[1]], requires_grad=False).to(device).type(dtype))
            loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
            if self.cfg.robustkl:
                loss_kl = torch.sqrt(1 + loss_kl**2)-1
            loss_kl *= weight_kl

            ################################# dist matrix l2 loss
            if 'dist_matrix' in drec.keys():
                target = dorig['dist_matrix'].to(device)
                loss_dist_matrix_rec = self.LossL1(drec['dist_matrix'], target)
                if 'contactness' in drec.keys():
                    loss_dist_matrix_rec = loss_dist_matrix_rec * dorig['contactness'] 
                    loss_dist_matrix_rec = loss_dist_matrix_rec.sum(dim=[1,2]) * (1.0 / dorig['contactness'].sum(dim=[1,2]))
                    loss_dist_matrix_rec = loss_dist_matrix_rec / target.shape[1]
                loss_dist_matrix_rec = weight_rec * torch.mean(loss_dist_matrix_rec)
            else:
                loss_dist_matrix_rec = torch.tensor(0.0).to(device)

            if 'joint_rot' in drec.keys():
                loss_joint_rot = self.LossL2(drec['joint_rot'], dorig['joint_rot'])
                loss_joint_rot = weight_rec * torch.mean(loss_joint_rot)
            else:
                loss_joint_rot = torch.tensor(0.0).to(device)

            if 'contactness' in drec.keys():
                loss_contact = F.binary_cross_entropy(drec['contactness'].squeeze(), dorig['contactness'].squeeze().float())
                loss_contact = loss_contact.mean()
            else:
                loss_contact = torch.tensor(0.0).to(device)
            ################################## loss dict
            loss_dict = {'loss_kl': loss_kl,
                        #  'loss_object_contact_rec': loss_object_contact_rec,
                        #  'loss_markers_contact_rec': loss_markers_contact_rec,
                        #  'loss_marker_rec': loss_marker_rec,
                        'loss_contact': loss_contact,
                        'loss_dist_matrix_rec': loss_dist_matrix_rec,
                        #  'loss_normal_rec': loss_normal_rec,
                        "loss_joint_rot": loss_joint_rot,
                        }
            loss_total = torch.stack(list(loss_dict.values())).sum()
            # loss_dict['loss_total'] = loss_total
            if (dorig['contactness'].sum(dim=[1,2]) == 0).any():
                for i, c in enumerate(dorig['contactness'].sum(dim=[1,2])):
                    if c == 0:
                        logger.warning(f"Warning: contactness is zero on {dorig['seq_name'][i]}")
    
            storage = get_event_storage()
            if comm.is_main_process():
                
                average_contactness_cnt =  dorig['contactness'].sum(dim=[1,2]).to(torch.float32).mean().item()
                storage.put_scalar('average_contactness_cnt', average_contactness_cnt)


            return loss_dict



