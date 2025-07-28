import torch
import logging
import logging

import d2.utils.comm as comm
from d2.engine.defaults import create_ddp_model, DefaultTrainer
from d2.solver import build_lr_scheduler, WarmupCosineLR
from d2.solver.build import get_default_optimizer_params
from tqdm import tqdm
from d2.engine.train_loop import HookBase

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


class MLDVaeTrainer(DefaultTrainer):
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
        from mld.models.architectures.mld_vae import MldVae
        model = MldVae(nfeats=cfg.MODEL.INPUT_DIM)
        
        model.to(cfg.DEVICE)
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


    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def __call__(self, data):
        x0 = {'feat': data['feat'].to(self.device)}
        y = {'clip': data['text_features'].to(self.device), 'bps': data['bps_object'].to(self.device)}

        result = self.model(x0['feat'], [x0['feat'].shape[1]]*x0['feat'].shape[0])
        
        loss = self.loss_net(x0['feat'], result)
        
        return loss


    def evaluate(self, dataloader):
        # object_contact_metrics = ObjectContactMetrics(self.device, False)
        self.model.eval()

        eval_loss_dict_net = {}
        cnt_vis = 0
        for dorig in tqdm(dataloader):
            with torch.no_grad():
                
                x0 = {'feat': dorig['feat'].to(self.device)}
                y = {'clip': dorig['text_features'].to(self.device), 'bps': dorig['bps_object'].to(self.device)}

                result = self.model(x0['feat'], [x0['feat'].shape[1]]*x0['feat'].shape[0])
                
                loss = self.loss_net(x0['feat'], result)
                eval_loss_dict_net = {k: eval_loss_dict_net.get(k, 0.0) + v.item() for k, v in loss.items()}

            cnt_vis = cnt_vis + 1

        eval_loss_dict_net = {f'eval/{k}': sum(comm.all_gather(v)) / (len(dataloader)*comm.get_world_size()) for k, v in eval_loss_dict_net.items()}
        self.model.train()
        return eval_loss_dict_net


    def loss_net(self, x0, out):
        device = x0.device
        dtype = x0.dtype
        feats_rst, z, dist = out
        # import ipdb; ipdb.set_trace()
        mean, std = dist.mean, dist.stddev
        q_z = dist
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros([mean.shape[0], mean.shape[1], mean.shape[2]], requires_grad=False).to(device).type(dtype),
            scale=torch.ones([mean.shape[0], mean.shape[1], mean.shape[2]], requires_grad=False).to(device).type(dtype))
        loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[2]))
        
        loss_recon = self.LossL2(x0, feats_rst).mean()

        loss_dict = {
            'loss_kl': loss_kl/mean.shape[2],
            'loss_recon': loss_recon
        }
        return loss_dict



