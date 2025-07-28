import json
import os.path as op
import sys

import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from easydict import EasyDict
from scipy.spatial.distance import cdist
from torch_geometric.nn.pool import fps

current_directory = op.dirname(op.abspath(__file__))
parent_directory = op.dirname(current_directory)
sys.path.append(parent_directory)
sys.path = [".."] + sys.path
import utils.arctic.common.thing as thing
from utils.arctic.common.rot import axis_angle_to_quaternion, quaternion_apply
from utils.arctic.common.object_tensors import ObjectTensors as ArcticObjectTensors

# objects to consider for training so far
GRAB_OBJS = ["airplane", "camera", "cylindermedium", "flashlight", "headphones", "piggybank", "spheremedium", "toothbrush", "watch",
        "alarmclock", "cubelarge", "cylindersmall", "flute", "knife", "pyramidlarge", "spheresmall", "toothpaste", "waterbottle", "apple",
        "cubemedium", "doorknob", "fryingpan", "lightbulb", "pyramidmedium", "stamp", "toruslarge", "wineglass", "banana", 
        "cubesmall", "duck", "gamecontroller", "mouse", "pyramidsmall", "stanfordbunny", "torusmedium", "binoculars", "cup", 
        "elephant", "hammer", "mug", "scissors", "stapler", "torussmall", "bowl", "cylinderlarge", "eyeglasses", "hand", "phone", "spherelarge", "teapot", "train"]


class GrabObjectTensors(nn.Module):
    def __init__(self, data_path='.', fps_target_size=2048):
        super(GrabObjectTensors, self).__init__()
        self.fps_target_size = fps_target_size
        self.objs = []
        path = os.path.join(data_path, 'obj')
        for obj_name in GRAB_OBJS:
            mesh = trimesh.load(
                os.path.join(path, f'{obj_name}.ply'), process=False
            )
            v = mesh.vertices 
            normal = mesh.vertex_normals
            f = mesh.faces
            self.objs.append(
                {
                    "v": torch.from_numpy(v),
                    "f": torch.from_numpy(f),
                    "normal": torch.from_numpy(normal),
                    "fps_idx": fps(torch.from_numpy(v), ratio=fps_target_size/v.shape[0], random_start=False).numpy()
                }
            )
            

    def redo_fps(self):
        num_objs = self.obj_tensors["v"].shape[0]
        for oid in range(num_objs):
            max_len = self.obj_tensors["v_len"][oid]
            v = self.obj_tensors["v"][oid][:max_len]
            top_parts = (self.obj_tensors["parts_ids"][oid][:max_len] == 1)
            v_top_ids = torch.arange(max_len)[top_parts]
            v_top = v[top_parts, :]
            v_bottom = (self.obj_tensors["parts_ids"][oid][:max_len] == 2)

    def forward(
        self,
        global_orient: (None, torch.Tensor),
        transl: (None, torch.Tensor),
        obj_name: (int, str)
    ):
        # self._sanity_check(angles, global_orient, transl, query_names, fwd_template)
        seq_len = global_orient.shape[0]

        # meta info
        if isinstance(obj_name, str):
            obj_idx = GRAB_OBJS.index(obj_name)
        else:
            obj_idx = obj_name
        v = [self.objs[obj_idx]["v"]] * seq_len
        v = torch.stack(v, dim=0)
        f = self.objs[obj_idx]['f']

        # articulation + global rotation
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))

        
        # collect entities to be transformed
        tf_dict = {}
        tf_dict["v"] = v.clone()
        # global rotation for all
        for key, val in tf_dict.items():
            val_rot = quaternion_apply(quat_global[:, None, :], val)
            if transl is not None:
                val_rot = val_rot + transl[:, None, :]
            tf_dict[key] = val_rot
        tf_dict["sub_v"] = tf_dict[:, self.objs[obj_idx]['fps_idx'], :]
        tf_dict['f'] = f
        return tf_dict


    def to(self, dev):
        self.objs = thing.thing2dev(self.objs, dev)
        self.dev = dev

