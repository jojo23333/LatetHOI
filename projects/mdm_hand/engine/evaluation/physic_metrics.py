import torch
import numpy as np
import smplx
from .metrics import ObjectContactMetrics

GRAB_OBJECTS = ['cylinderlarge', 'mug', 'elephant', 'hand', 'cubelarge', 'stanfordbunny', 'airplane', 'alarmclock', 'banana', 'body', 'bowl', 'cubesmall', 'cup', 'doorknob', 'cubemedium',
             'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer', 'headphones', 'knife', 'lightbulb', 'mouse', 'phone', 'piggybank', 'pyramidlarge', 'pyramidsmall', 'pyramidmedium',
             'duck', 'scissors', 'spherelarge', 'spheresmall', 'stamp', 'stapler', 'table', 'teapot', 'toruslarge', 'torussmall', 'train', 'watch', 'cylindersmall', 'waterbottle', 'torusmedium',
             'cylindermedium', 'spheremedium', 'wristwatch', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste', 'apple', 'toothbrush']

ARCTIC_OBJECTS = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    "scissors",
    "phone",
]

class PhysicEvaluation:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.device = 'cpu'#self.cfg.DEVICE
        self.contact_collision_metric = ObjectContactMetrics(self.device, False)
        
    def compute_all_metrics(self, _vertices_sbj, _faces_sbj, _verts_obj, _faces_obj, eval_n_frames=-1, device=None) -> dict:
        if eval_n_frames == -1:
            v_sbj = _vertices_sbj
            v_obj = _verts_obj
        else:
            frames = np.linspace(start=0, stop=len(_verts_obj) - 1, num=eval_n_frames).round()
            v_sbj = _vertices_sbj[frames]
            v_obj = _verts_obj[frames]
        other_info = self.contact_collision_metric.compute_metrics(
            sbj_vertices=v_sbj, sbj_faces=_faces_sbj,
            obj_vertices=v_obj, obj_faces=_faces_obj
        )
        volume_pred = self.contact_collision_metric.volume
        depth_pred = self.contact_collision_metric.depth
        contact_ratio_pred = self.contact_collision_metric.contact_ratio
        jerk = self.contact_collision_metric.jerk
        return {**volume_pred, **depth_pred, **contact_ratio_pred, **jerk}, other_info


    def __call__(self, obj_v, obj_f, rhand_v, rhand_f, lhand_v=None, lhand_f=None, eval_n_frames=50):
        
        if isinstance(obj_v, np.ndarray):
            obj_v = torch.from_numpy(obj_v).to(self.device).to(torch.float32)

        if isinstance(obj_f, np.ndarray):
            obj_f = torch.from_numpy(obj_f.astype(np.int32)).to(self.device)
        
        if isinstance(rhand_v, np.ndarray):
            rhand_v = torch.from_numpy(rhand_v).to(self.device).to(torch.float32)
        
        if isinstance(rhand_f, np.ndarray):
            rhand_f = torch.from_numpy(rhand_f.astype(np.int32)).to(self.device)
        
        if lhand_v is not None and isinstance(lhand_v, np.ndarray):
            lhand_v = torch.from_numpy(lhand_v).to(self.device).to(torch.float32)
        
        if lhand_f is not None and isinstance(lhand_f, np.ndarray):
            lhand_f = torch.from_numpy(lhand_f.astype(np.int32)).to(self.device)

        res_dict_r, other_info_r = self.compute_all_metrics(
            rhand_v, rhand_f,
            obj_v, obj_f,
            device=self.device,
            eval_n_frames=eval_n_frames,
        )

        if lhand_v is not None:
            res_dict_l, other_info_l = self.compute_all_metrics(
                lhand_v, lhand_f,
                obj_v, obj_f,
                device=self.device,
                eval_n_frames=eval_n_frames
            )
            volume_list_r, depth_list_r, ratio_list_r, off_ground_masks_r = other_info_r
            volume_list_l, depth_list_l, ratio_list_l, off_ground_masks_l = other_info_l
            in_contact_ratio = torch.tensor([x+y for x,y in zip(ratio_list_l, ratio_list_r)])
            in_contact_flag = (in_contact_ratio > 0.02).to(torch.int)
            off_ground_masks_r = off_ground_masks_r > 0
            off_ground_contact_ratio = torch.masked_select(in_contact_ratio, off_ground_masks_r).sum() / off_ground_masks_r.sum()
            off_ground_contact_rate = torch.masked_select(in_contact_flag, off_ground_masks_r).sum() / off_ground_masks_r.sum()
        else:
            volume_list_r, depth_list_r, ratio_list_r, off_ground_masks_r = other_info_r
            in_contact_ratio = torch.tensor(ratio_list_r)
            in_contact_flag = (in_contact_ratio > 0.02).to(torch.int)
            off_ground_masks_r = off_ground_masks_r > 0
            off_ground_contact_ratio = torch.masked_select(in_contact_ratio, off_ground_masks_r).sum() / off_ground_masks_r.sum()
            off_ground_contact_rate = torch.masked_select(in_contact_flag, off_ground_masks_r).sum() / off_ground_masks_r.sum()
            res_dict_l = {}

        return {"rhand": res_dict_r, "lhand": res_dict_l, "off_ground_contact_ratio": off_ground_contact_ratio.item(), "off_ground_contact_rate": off_ground_contact_rate.item()}

import argparse, os, json
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script with command line arguments")
    parser.add_argument(
        '-f', "--file_path", type=str, required=True, help="The file path argument"
    )
    parser.add_argument(
        '-n', "--eval_n_frames", type=int, default=50, help="The file path argument"
    )
    parser.add_argument("--smooth", action='store_true')
    args = parser.parse_args()
    eval = PhysicEvaluation()
    
    assert os.path.exists(args.file_path), "File do not exist"
    assert args.file_path.endswith('.pth'), "Wrong format"
    export_result_path = args.file_path.replace('.pth', '.json')
    os.makedirs(
        os.path.join(
            os.path.dirname(export_result_path), 'eval'
        ), exist_ok=True
    )
    export_result_path = os.path.join(
        os.path.dirname(export_result_path), 'eval', os.path.basename(export_result_path)
    )
    print("Metric goes to:", export_result_path)
    data = torch.load(args.file_path, map_location='cpu')
    if isinstance(data, dict):
        data = data['pred']
    obj_v, obj_f, rhand_v, rhand_f, lhand_v, lhand_f = data
    print(f"Starting the evaluation for seq {args.file_path}")
    result = eval(obj_v, obj_f, rhand_v, rhand_f, lhand_v, lhand_f, eval_n_frames=args.eval_n_frames)
    print(result)
    with open(export_result_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    
    