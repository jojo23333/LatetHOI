# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
import weakref
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

# import d2.data.transforms as T
from d2.engine.checkpoints import GeneralCheckpointer
from d2.config import CfgNode, LazyConfig
from d2.utils.events import EventStorage, get_event_storage

# from d2.modeling import build_model
# from d2.solver import build_lr_scheduler, build_optimizer
from d2.utils import comm
from d2.utils.collect_env import collect_env_info
from d2.utils.env import seed_all_rng
from d2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from d2.utils.file_io import PathManager
from d2.utils.logger import setup_logger

from torch.utils.data import DataLoader

from d2.data.common import worker_init_reset_seed
from d2.data.build import build_dataset

from . import hooks
from .train_loop import AMPTrainer, SimpleTrainer, TrainerBase

__all__ = [
    "create_ddp_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultPredictor",
    "DefaultTrainer",
]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
            Examples:

            Run on single machine:
                $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

            Change some config options:
                $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

            Run on multiple machines:
                (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
                (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read()
                # _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            # logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            logger.info("Running with full config:\n{}".format(cfg.dump(), ".yaml"))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]



class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg, custom_hooks=[], ModelWrapper=None, is_train=True):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        self.cfg = cfg
        self.logger = logging.getLogger('d2')
        if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        # TODO fix following line
        # cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.logger.info("Building DDP")
        model = model.to(cfg.DEVICE)
        self.model = create_ddp_model(model, broadcast_buffers=False)
        if ModelWrapper is None:
            self.logger.warning("No ModelWrapper is provided, model will not be wrapped")
            self.wrapped_model = None
        else:
            self.wrapped_model = ModelWrapper(self.model, cfg, device=cfg.DEVICE)

        if is_train:
            self.logger.info("Building Dataloader")
            self.train_loader, sampler = self.build_dataset_loader(self.cfg, is_train=True)
            self.optimizer = self.build_optimizer(cfg, model)
            self.logger.info("Building Trainer")
            self._trainer = (AMPTrainer if self.cfg.AMP_ENABLED else SimpleTrainer)(
                self.wrapped_model, self.train_loader, self.optimizer, async_write_metrics=True, data_sampler=sampler,
            )
            self._trainer.register_hooks(custom_hooks)
            # Build LR scheduler
            self.logger.info("Max Epoch: {}".format(cfg.SOLVER.MAX_EPOCH))
            self.logger.info("It per Epoch: {}".format(len(self.train_loader)))
            self.max_iter = cfg.SOLVER.MAX_EPOCH * len(self.train_loader) if cfg.SOLVER.MAX_EPOCH > 0 else cfg.SOLVER.MAX_ITER
            self.cfg.defrost()
            self.cfg.SOLVER.MAX_ITER = self.max_iter
            self.cfg.freeze()
            self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
            print(comm.get_local_rank(), "Finish building")

        self.start_iter = 0
        
        self.logger.info(f"Max Iteration: {self.max_iter}")
        self.cfg = cfg
        self.checkpointer = GeneralCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.register_hooks(self.build_hooks()+custom_hooks)

    def build_model(self, cfg):
        raise NotImplementedError

    def build_optimizer(self, cfg):
        raise NotImplementedError

    def build_train_loader(self, cfg):
        raise NotImplementedError
    
    def build_lr_scheduler(self, cfg, optimizer):
        raise NotImplementedError
    

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

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            # hooks.PreciseBN(
            #     # Run at the same freq as (but before) evaluation.
            #     cfg.TEST.EVAL_PERIOD,
            #     self.model,
            #     # Build a new data loader to not affect training
            #     self.build_train_loader(cfg),
            #     cfg.TEST.PRECISE_BN.NUM_ITER,
            # )
            # if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            # else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=cfg.SOLVER.CKPT_KEEP_NUM))

        def test_and_save_results():
            self._last_eval_results = self.evaluate()
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        
        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            from d2.engine.defaults import default_writers
            writers = default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
            ret.append(hooks.PeriodicWriter(writers, period=20))
        return ret

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        self.logger.info("Starting training from iteration {}".format(self.start_iter))
        self.logger.info("Max Iteration: {}".format(self.max_iter))
        self.iter = self.start_iter

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                self.logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
 
    def benchmark(self):
        from torch.profiler import profile, record_function, ProfilerActivity
        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            print(output)
            p.export_chrome_trace(os.path.join(self.cfg.OUTPUT_DIR, "benchmark_s"+str(p.step_num) + ".json"))

        NUM_TEST_EPOCH = 10
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
        ) as p:
            with EventStorage(self.start_iter) as self.storage:
                try:
                    self.before_train()
                    for i in range(0, NUM_TEST_EPOCH):
                        self.logger.info(f"SPEED TEST: Running iteration {i}/{NUM_TEST_EPOCH}")
                        self.before_step()
                        self.run_step()
                        self.after_step()
                        p.step()
                    # self.iter == max_iter can be used by `after_train` to
                    # tell whether the training successfully finished or failed
                    # due to exceptions.
                    self.iter += 1
                except Exception:
                    self.logger.exception("Exception during training:")
                    raise
        
 
    def run_step(self):
        # TODO handle one's own run_step
        raise NotImplementedError
        # self._trainer.iter = self.iter
        # self._trainer.run_step()

    def state_dict(self):
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    def evaluate(self):
        raise NotImplementedError

    def get_model_info(self, model):
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn.parameter_count import parameter_count
        num_params = parameter_count(model)
        self.logger.info(f"Prameter Count: {num_params['']/1e6}M")

    def build_dataset_loader(self, cfg, is_train=True):
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
            drop_last=is_train
            # shuffle=sampler
        )
        return loader, sampler

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        For example, with the original config like the following:

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


# # Access basic attributes from the underlying trainer
# for _attr in ["model", "data_loader", "optimizer"]:
#     setattr(
#         DefaultTrainer,
#         _attr,
#         property(
#             # getter
#             lambda self, x=_attr: getattr(self._trainer, x),
#             # setter
#             lambda self, value, x=_attr: setattr(self._trainer, x, value),
#         ),
#     )
