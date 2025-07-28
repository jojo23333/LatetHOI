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
from datetime import datetime
from engine.default_mld_vae import MLDVaeTrainer
from engine.default_mld_ldm import MLDLdmTrainer
import logging
logging.basicConfig(level=logging.INFO)

from datasets import *

DEBUG = True

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.OUTPUT_DIR == '':
        cfg.OUTPUT_DIR = os.path.join('.exps', os.path.basename(args.config_file).split('.')[0])
        current_date = datetime.now().strftime("%m_%d")
        cfg.OUTPUT_DIR = os.path.join(
            os.path.dirname(cfg.OUTPUT_DIR),
            f"{current_date}_{os.path.basename(cfg.OUTPUT_DIR)}"
        )
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        cfg.defrost()
        cfg.AMP_ENABLED = False
        cfg.DATASET.NUM_WORKERS = 0
        cfg.TEST.EVAL_PERIOD = 10
        cfg.TRAINING.BATCH_SIZE = 4
        cfg.OUT_DIR = './.exps/debug'
    
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
        cfg.defrost()
        cfg.AMP_ENABLED = False
        cfg.DATASET.NUM_WORKERS = 0
        cfg.TEST.EVAL_PERIOD = 10
        cfg.OUT_DIR = './.exps/debug'
    if args.eval_only:
        cfg.defrost()
        cfg.AMP_ENABLED = False
        cfg.freeze()
        trainer = MLDLdmTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.evaluate(eval_mode=True)
    else:
        if args.mode == 'vae':
            trainer = MLDVaeTrainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            trainer.train()
        else:
            trainer = MLDLdmTrainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = default_argument_parser()
    parser.add_argument("-m", "--mode", type=str, default="vae", help="mdm or ldm")
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
