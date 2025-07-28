import torch
import logging
import weakref
import logging
import os
import json
from typing import Any

import d2.utils.comm as comm
from d2.engine.checkpoints import GeneralCheckpointer
from d2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from d2.engine.defaults import create_ddp_model
from d2.data.samplers import TrainingSampler, InferenceSampler
from d2.data.common import ToIterableDataset, worker_init_reset_seed
from d2.data.build import build_dataset
from d2.solver import build_lr_scheduler
from d2.solver.build import get_default_optimizer_params
from d2.utils.logger import setup_logger
from d2.engine import hooks

from torch.utils.data import DataLoader
from modeling.build import create_model

from engine.evaluation.eval_generation import EvalGeneration
from engine.model_wrapper import MANODiffusionWrapper

DEBUG = False

class DiffusionTrainer(TrainerBase):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        self.logger = logging.getLogger('d2')
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        # Assume these objects must be constructed in this order.
        self.cfg = cfg
        model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, model)
        self.model = create_ddp_model(model, broadcast_buffers=False)

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.checkpointer = GeneralCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        # diffusion
        self.diffusion_model = MANODiffusionWrapper(cfg, model)
        self.logger.info("Building Trainer")
        self.evaluators = self.build_evaluators()
        self.register_hooks(self.build_hooks())
        print(comm.get_local_rank(), "Finish building")
        self.train_loader = self.build_dataset_loader(self.cfg, self.optimizer)
        self._trainer = (AMPTrainer if self.cfg.AMP_ENABLED else SimpleTrainer)(
            self.diffusion_model, self.train_loader, self.optimizer, async_write_metrics=True
        )

    def build_optimizer(self, cfg, model):
        """
            NOTE: Implement your optimizer here
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
        )
        return torch.optim.AdamW(
            params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

    def build_lr_scheduler(self, cfg, optimizer):
        """
            NOTE: Implement your lr scheduler here
        """
        return build_lr_scheduler(cfg, optimizer)

    def build_model(self, cfg):
        """
            NOTE build your model here
        """
        model = create_model(cfg)
        model.to(cfg.DEVICE)
        if comm.is_main_process():
            self.get_model_info(model)
        return model

    def get_model_info(self, model):
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn.parameter_count import parameter_count
        # flops = FlopCountAnalysis(
        #     model,
        #     (
        #         torch.randn((2, 60, 58)).to(self.cfg.DEVICE),
        #         torch.randn((2, 1, 1)).to(self.cfg.DEVICE),
        #         None,
        #         True
        #     )
        # )
        num_params = parameter_count(model)
        # self.logger.info(f"FLOPs Count: {flops.total()/1e9}")
        # self.logger.info(str(flops.by_operator()))
        self.logger.info(f"Prameter Count: {num_params['']/1e6}M")
        # self.logger.info(str(num_params))

    def build_dataset_loader(self, cfg, is_train=True):
        """
            NOTE build your dataloader here
        """
        num_workers = cfg.DATASET.NUM_WORKERS

        if is_train:
            batch_size = cfg.TRAINING.BATCH_SIZE
            dataset = build_dataset(cfg, "train")
            if comm.get_world_size() == 1:
                sampler = None
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
            dataset = build_dataset(cfg, cfg.TEST.SPLIT)
            # dataset = ToIterableDataset(
            #     dataset, InferenceSampler(len(dataset)), shard_sampler=True, shard_chunk_size=batch_size
            # )
            if comm.get_world_size() == 1:
                sampler = None
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            worker_init_fn=worker_init_reset_seed,
            persistent_workers=num_workers > 0,
            shuffle=sampler is None,
            collate_fn=dataset.collate_fn
        )

        return loader

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.logger.info("Start Training")
        super().train(self.start_iter, self.max_iter)

    def evaluate(self, vis=False, write_metric=True):
        self.logger.info("Begin Evaluating")
        if "test_loader" not in self.__dict__:
            self.test_loader = self.build_dataset_loader(self.cfg, is_train=False)

        writable_metrics = {}
        for evaluator in self.evaluators:
            result = evaluator(self.iter, test_loader=self.test_loader, vis=vis, train_loader=self.train_loader)
            for k, v in result.items():
                writable_metrics[f'{evaluator.__class__.__name__}_{k}'] = v
        if write_metric and comm.is_main_process():
            with open(os.path.join(self.cfg.OUTPUT_DIR, 'result.json'), 'w') as f:
                json.dump(writable_metrics, f)
        return writable_metrics

    def run_step(self):
        if "_trainer" not in self.__dict__:
            self.train_loader = self.build_dataset_loader(self.cfg, self.optimizer)
            self._trainer = (AMPTrainer if self.cfg.AMP_ENABLED else SimpleTrainer)(
                self.diffusion_model, self.train_loader, self.optimizer, async_write_metrics=True
            )

        self._trainer.iter = self.iter
        self._trainer.run_step()

    def state_dict(self):
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "_trainer" in self.__dict__:
            self._trainer.load_state_dict(state_dict["_trainer"])
    
    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_evaluators(self):
        evaluators = []
        # if self.cfg.DATASET.NAME == 'a101':
        #     evaluators.append(PoseRefineEvaluator(self.cfg, self.diffusion_model))
        # evaluators.append(InpaintEvaluator(self.cfg, self.diffusion_model))
        evaluators.extend([
            # EvalGeneration(self.cfg, self.diffusion_model, 'train'),
            EvalGeneration(self.cfg, self.diffusion_model, 'val'),
            # EvalGeneration(self.cfg, self.diffusion_model, 'test')
        ])
        return evaluators

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler()
            # NOTE considering adding precise bn later of bn is used
            # Do PreciseBN before checkpointer, because it updates the model and need to
            # be saved by checkpointer.
            # This is not always the best: if checkpointing has a different frequency,
            # some checkpoints may have more precise statistics than others.
        ]

        if comm.is_main_process():
            # frequent checkpointer
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=10
                ),
            )

        def test_and_save_results():
            self._last_eval_results = self.evaluate()
            return self._last_eval_results

        # def test_and_vis_results():
        #     self._last_eval_results = self.evaluate(vis=True)
        #     return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results))
        # ret.append(hooks.EvalHook(self.cfg.TEST.VIS_PERIOD, self.vis))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            from d2.engine.defaults import default_writers
            writers = default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
            ret.append(hooks.PeriodicWriter(writers, period=20))

        return ret
