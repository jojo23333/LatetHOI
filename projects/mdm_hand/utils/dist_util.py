"""
Helpers for distributed training.

This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
Modified from detectron2(https://github.com/facebookresearch/detectron2)

Copyright (c) Xiaoyang Wu (xiaoyang.wu@connect.hku.hk). All Rights Reserved.
Please cite our work if you use any part of the code.
"""
import os
import functools
import numpy as np
import torch
import socket
import logging
import subprocess
import torch.distributed as dist
import torch.multiprocessing as mp

from datetime import timedelta
from utils import comm

DEFAULT_TIMEOUT = timedelta(minutes=10)


def get_dist_setting(args):
    # for slurm, launch 1 process per machine
    if 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ.get('SLURM_NTASKS', 1))
        # ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        port = os.environ.get('MASTER_PORT', '29500')

        num_gpus_per_machine = torch.cuda.device_count()
        num_machines = ntasks
        machine_rank = proc_id
        dist_url = f"tcp://{addr}:{port}"
        return num_gpus_per_machine, num_machines, machine_rank, dist_url
    else:
        return args.num_gpus, args.num_machines, args.machine_rank, args.dist_url


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    cfg=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
        
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                cfg,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*cfg)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    cfg,
    timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group.
    comm.create_local_process_group(num_gpus_per_machine)
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*cfg)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available() and torch.cuda.current_device()>=0:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return torch.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

