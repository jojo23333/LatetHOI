import torch
from torch.utils.data import DataLoader
from datasets.tensors import collate as all_collate
from datasets.tensors import t2m_collate
from utils.seed import init_worker_with_seed
from functools import partial
from utils import comm

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from datasets.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from datasets.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from datasets.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

def get_dataset_loader_a101(batch_size, split='train', num_workers=4, seed=None):
    from datasets.a101.dataset import Dataset as A101
    from datasets.a101.config import base_cfg as a101_cfg
    from torchvision.transforms import transforms
    from datasets.a101.utils.transforms import normalize_pose_xyz

    dataset = A101(transforms.ToTensor(), a101_cfg, split, transforms=[normalize_pose_xyz])
    if comm.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    init_fn = partial(
        init_worker_with_seed, num_workers=num_workers, rank=comm.get_rank(),
        seed=seed) if seed is not None else None

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        # pin_memory=True,
        worker_init_fn=init_fn,
        persistent_workers=num_workers>0
    )
    return loader

def build_transform(cfg):
    from datasets.transforms import RandomRotate
    transforms = []
    rotate_x = cfg.DATASET.AUG.ROTATE_X
    rotate_y = cfg.DATASET.AUG.ROTATE_Y
    rotate_z = cfg.DATASET.AUG.ROTATE_Z
    if rotate_x[0] > 0:
        transforms.append(RandomRotate(axis='z', angle=[rotate_z[1], rotate_z[2]], p=rotate_z[0]))
    if rotate_x[0] > 0:
        transforms.append(RandomRotate(axis='x', angle=[rotate_x[1], rotate_x[2]], p=rotate_x[0]))
    if rotate_y[0] > 0:
        transforms.append(RandomRotate(axis='y', angle=[rotate_y[1], rotate_y[2]], p=rotate_y[0]))
    return transforms

def get_dataset(cfg, split='train'):
    if cfg.DATASET.NAME == 'a101':
        return get_a101_dataset(cfg, split)
    elif cfg.DATASET.NAME == 'interhand':
        return get_interhand_dataset(cfg, split)
    elif cfg.DATASET.NAME == 'hand_motion':
        return get_hand_motion_dataset(cfg, split)
    elif cfg.DATASET.NAME.startswith("arctic"):
        return get_hand_mano_motion_dataset(cfg, split)
    else:
        raise NotImplementedError

def get_hand_mano_motion_dataset(cfg, split):
    from datasets.hand_mano_motion_dataset import SingleHandDataset, ManoMotionDataset
    from datasets.transforms import to_tensor
    num_frames = cfg.DATASET.NUM_FRAMES
    fps = cfg.DATASET.FPS
    name = cfg.DATASET.NAME
    dataset = ManoMotionDataset(cfg, fps, num_frames, num_frames, datasets=[name], mode=split, transforms=[to_tensor])
    return dataset

def get_hand_motion_dataset(cfg, split):
    from datasets.hand_motion_dataset import HandMotionDataset
    from datasets.transforms import to_tensor
    num_frames = cfg.DATASET.NUM_FRAMES
    fps = cfg.DATASET.FPS
    dataset = HandMotionDataset(fps, num_frames, num_frames, datasets=['h2o', 'a101', 'arctic'], mode=split, transforms=[to_tensor])
    return dataset

def get_interhand_dataset(cfg, split):
    from datasets.interhand.dataset_motion import InterHandMotion
    from datasets.transforms import to_tensor
    num_frames = cfg.DATASET.NUM_FRAMES
    fps = cfg.DATASET.FPS
    sample_gap = 30 // fps
    return InterHandMotion([to_tensor], split, num_frames, sample_gap)

def get_a101_dataset(cfg, split='train'):
    from datasets.a101.dataset import Dataset as A101
    from torchvision.transforms import transforms
    from datasets.a101.config import base_cfg as a101_cfg
    from datasets.a101.utils.transforms import normalize_pose_xyz
    from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp.add_function(A101.__init__)
    # lp.enable_by_count()
    # NOTE: Moving transforms to GPU for less cpu overhead
    transforms = []
    return A101(cfg, a101_cfg, split, transforms=transforms)