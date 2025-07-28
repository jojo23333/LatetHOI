import random
from typing import Any
import numpy as np
import torch

def to_tensor(d):
    if isinstance(d, dict):
        return {k: to_tensor(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d)
    else:
        return d

class RandomRotate(object):
    def __init__(self,
                 angle=None,
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            print("Not implemented")
            raise NotImplementedError
        if "joint_world" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["joint_world"].min(axis=(0, 1))
                x_max, y_max, z_max = data_dict["joint_world"].max(axis=(0, 1))
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["joint_world"] -= center
            data_dict["joint_world"] = np.dot(data_dict["joint_world"], np.transpose(rot_t))
            data_dict["joint_world"] += center
        del rot_t
        del angle
        return data_dict

class BatchRandomNoise(object):
    def __init__(self, noise_scale=2):
        self.scale = noise_scale
    
    def __call__(self, batch_data_dict) -> Any:
        batch_joints = batch_data_dict["joint_world"]
        B, T, N, C = batch_joints.shape
        batch_data_dict["joint_world"] = batch_joints + torch.randn((B, 1, N, C)).to(batch_joints) * self.scale
        return batch_data_dict

class BatchRandomRotate(object):
    def __init__(self, angle=None, center=None, axis='z', p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.p = p 
        self.center = None if center is None else torch.tensor(center).unsqueeze(0)

    def __call__(self, batch_data_dict):
        batch_joints = batch_data_dict["joint_world"]
        bsz, num_f, num_j, _ = batch_joints.shape
        mask = torch.rand([bsz], device=batch_joints.device) < self.p
        angles = (torch.rand([bsz,], device=batch_joints.device) * (self.angle[1]-self.angle[0]) - self.angle[0]) * torch.pi
        rot_cos, rot_sin = torch.cos(angles), torch.sin(angles)
        rot_t = torch.zeros([bsz, 3, 3], dtype=batch_joints.dtype, device=batch_joints.device)
        if self.axis == 'x':
            rot_t[:, 0, 0] = 1
            rot_t[:, 1, 1] = rot_cos
            rot_t[:, 1, 2] = -rot_sin
            rot_t[:, 2, 1] = rot_sin
            rot_t[:, 2, 2] = rot_cos
            #rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t[:, 0, 0] = rot_cos
            rot_t[:, 0, 2] = rot_sin
            rot_t[:, 1, 1] = 1
            rot_t[:, 2, 0] = -rot_sin
            rot_t[:, 2, 2] = rot_cos
            # rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t[:, 0, 0] = rot_cos
            rot_t[:, 0, 1] = -rot_sin
            rot_t[:, 1, 0] = rot_sin
            rot_t[:, 1, 1] = rot_cos
            rot_t[:, 2, 2] = 1
            # rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            print("Not implemented")
            raise NotImplementedError
        rot_t = rot_t.unsqueeze(1).repeat(1, num_f*num_j, 1, 1).view(-1, 3, 3)

        if self.center is None:
            c_min = batch_joints.amin(dim=(1, 2))
            c_max = batch_joints.amax(dim=(1, 2))
            center = torch.stack([(c_min[:, 0] + c_max[:, 0]) / 2, 
                                  (c_min[:, 1] + c_max[:, 1]) / 2, 
                                  (c_min[:, 2] + c_max[:, 2]) / 2], dim=-1)
            center = center.unsqueeze(1).unsqueeze(2)
        else:
            center = self.center
        batch_joints -= center
        batch_joints = batch_joints.view(-1, 1, 3)
        batch_joints = torch.bmm(batch_joints, rot_t.transpose(1, 2))
        batch_joints = batch_joints.view(bsz, num_f, num_j, 3) + center
        batch_data_dict["joint_world"][mask, ...] = batch_joints[mask, ...]
        
        return batch_data_dict