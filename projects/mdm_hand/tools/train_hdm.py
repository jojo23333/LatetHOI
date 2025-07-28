# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import sys
import os
import torch
from utils.dist_util import get_dist_setting
from d2.engine.launch import launch
from d2.engine.defaults import default_argument_parser, default_setup
from config.defaults import get_cfg
# from utils import comm
from engine.default import DiffusionTrainer
import logging
logging.basicConfig(level=logging.INFO)

DEBUG = True

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# TODO set model & data ddp
def main(args):
    ###############################################################################
    # TODO dirty hack to move around '.pdbhistory'
    if DEBUG:
        from IPython import get_ipython
        from ipdb.__main__ import _get_debugger_cls
        import ipdb
        debugger = _get_debugger_cls()
        shell = get_ipython()
        shell.debugger_history_file = os.path.join('./', '.pdbhistory')
    ###############################################################################
    cfg = setup(args)
    # print("Launching ", comm.get_local_rank())
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    if args.eval_only:
        cfg.defrost()
        cfg.AMP_ENABLED = False
        cfg.freeze()
        trainer = DiffusionTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.evaluate(vis=True, write_metric=True)
    else:
        trainer = DiffusionTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    num_gpus, num_machines, machine_rank, dist_url = get_dist_setting(args)
    num_gpus = min(args.num_gpus, num_gpus)
    print(dist_url, machine_rank, num_machines, num_gpus)
    launch(
        main,
        num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        args=(args,),
    )
