import torch
import numpy as np
import torch
import os

from smplx import MANO
from loguru import logger
from aitviewer.configuration import CONFIG as C
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.renderables.meshes import Meshes
from aitviewer.headless import HeadlessRenderer
from .object_tensors import GrabObjectTensors, ArcticObjectTensors
from utils.arctic.common.mesh import Mesh
from utils.visualize.aitviewer import construct_viewer_meshes, construct_hand_skeleton, ViewerData
from utils.visualize.viewer import Viewer

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog
except ImportError:
    from PyQt5.QtWidgets import QApplication, QFileDialog
    print("PyQt6 not found, imported PyQt5 instead.")


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

class HandMotionVisualizer:
    # root would be the base dir of project
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    def __init__(
        self,
        render_types=["rgb", "depth", "mask"],
        interactive=True,
        size=(2024, 2024),
        flat_hand_mean=True,
        dataset='arctic'
    ):
        MANO_MODEL_DIR = os.path.join(self.ROOT, 'data/ARCTIC/body_models/mano')
        print(MANO_MODEL_DIR)

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = {
            "right": MANO(
                MANO_MODEL_DIR,
                create_transl=True,
                use_pca=False,
                flat_hand_mean=flat_hand_mean,
                is_rhand=True,
            ),
            "left": MANO(
                MANO_MODEL_DIR,
                create_transl=True,
                use_pca=False,
                flat_hand_mean=flat_hand_mean,
                is_rhand=False,
            ),
            "arctic_object": ArcticObjectTensors(base_dir=self.ROOT),
            "grab_object": GrabObjectTensors(os.path.join(self.ROOT, f"data/motion/grab"))
        }
        for layer in self.layers.values():
            layer.to(self.dev)
        self.dataset = dataset

        if not interactive:
            v = HeadlessRenderer()
        else:
            v = Viewer(size=size, open_file_handler=self.open_file)

        self.v = v
        self.interactive = interactive
        self.render_types = render_types

        self.gui_initialized = False

    def view_interactive(self):
        self.v.run()

    def view_fn_headless(self, vid_p):
        v = self.v
        v._init_scene()
        if not os.path.exists(os.path.dirname(vid_p)):
            os.makedirs(os.path.dirname(vid_p))

        logger.info("Rendering to video" + vid_p)
        v.save_video(video_dir=vid_p)

    def check_format(self, batch):
        meshes_all, data = batch
        assert isinstance(meshes_all, dict)
        assert len(meshes_all) > 0
        for mesh in meshes_all.values():
            assert isinstance(mesh, Meshes)
        assert isinstance(data, ViewerData)

    # def render_seq(self, batch, out_folder="./render_out"):
    #     meshes_all, data = batch
    #     self.v.reset()
    #     self.setup_viewer(data)
    #     for mesh in meshes_all.values():
    #         self.v.scene.add(mesh)
    #     if not self.gui_initialized:
    #         self.gui_initialized = True
    #         if self.interactive:
    #             self.view_interactive()
    #         else:
    #             # num_iter = data["num_frames"]
    #             self.view_fn_headless(out_folder+'/test.mp4')

    def setup_viewer(self):
        v = self.v
        fps = 10
        # camera.show_path()
        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False
        v.scene.floor.position[1] = 3
        v.scene.camera.position = np.array((0.0, -1.0, 0))
        self.v = v

    def open_file(self):
        dialog = QFileDialog()
        file_name, _ = dialog.getOpenFileName(caption='Select file', directory='/Volumes/graphics/nimble/users/muchenli/EXP_OUTPUT/hdm')
        if file_name.endswith('npy'):
            data = np.load(file_name, allow_pickle=True).item()
        else:
            data = torch.load(file_name, map_location='cpu')
        print("Opening: ", file_name)
        self.render_seq(data)

    def get_hand_skeleton(self, batch, seq_name='', glob_viewer_rot=None, glob_viewer_trans=None):
        skeletons = [
            construct_hand_skeleton(
                batch['pos_l'] - glob_viewer_trans,
                color='blue' if seq_name == 'gt' else 'red',
                name=f"{seq_name}_left_skeleton",
                glob_rot=glob_viewer_rot
            ),
            construct_hand_skeleton(
                batch['pos_r'] - glob_viewer_trans,
                color='blue' if seq_name == 'gt' else 'red',
                name=f"{seq_name}_right_skeleton",
                glob_rot=glob_viewer_rot
            )
        ]
        return skeletons


    def grab_obj_mesh(self, batch, seq_name, global_trans):
        if 'rot_obj' in batch and 'type_obj' in batch:
            with torch.no_grad():
                obj_out = self.layers["grab_object"](
                    global_orient=batch["rot_obj"],
                    transl=batch["trans_obj"],
                    obj_name=batch['type_obj'] 
                )
                vo = obj_out["v"]
                f3d_o = obj_out["f"]
                return {
                    "obj" :{
                        "v3d": vo.cpu().numpy() - global_trans,
                        "f3d": f3d_o.cpu().numpy(),
                        "vc": None,
                        "name": f"{seq_name}_object",
                        "color": "cyan-light" if seq_name =='gt' else "light-blue",
                    }
                }
        else:
            return {}

    def arctic_obj_mesh(self, batch, seq_name, global_trans):
        if 'rot_obj' in batch and 'type_obj' in batch:
            with torch.no_grad():
                nf = batch["rot_obj"].shape[0]
                obj_name = batch['type_obj'] if isinstance(batch['type_obj'], str) else ARCTIC_OBJECTS[batch['type_obj']]
                if not isinstance(obj_name ,str):
                    obj_name = ARCTIC_OBJECTS[obj_name]
                query_names = [obj_name] * nf
                obj_out = self.layers["arctic_object"](
                    angles=batch["arti_obj"].view(-1, 1),
                    global_orient=batch["rot_obj"],
                    transl=batch["trans_obj"],
                    query_names=query_names
                )
                vo = obj_out["v"]
                f3d_o = Mesh(
                    filename=os.path.join(self.ROOT, f"data/ARCTIC/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj")
                ).f
                return {
                    "obj" :{
                        "v3d": vo.cpu().numpy() - global_trans,
                        "f3d": f3d_o,
                        "vc": None,
                        "name": f"{seq_name}_object",
                        "color": "cyan-light" if seq_name =='gt' else "light-blue",
                    }
                }
        else:
            return {}

    def arctic_hand_mesh(self, batch, seq_name, global_trans, is_right=True):
        if is_right:
            nf = batch["rot_r"].shape[0]
            shape_r = batch["shape_r"]
            if shape_r.ndim == 1:
                shape_r = shape_r.unsqueeze(0).repeat(nf, 1)
            right_v = self.layers["right"](
                global_orient=batch['rot_r'].to(torch.float32),
                hand_pose=batch['pose_r'].to(torch.float32),
                betas=shape_r.to(torch.float32)
            )
            right_v = right_v.vertices.detach().cpu().numpy() - right_v.joints[:, 0:1, :].detach().cpu().numpy()
            right_v = right_v + batch["trans_r"].view(nf, 1, 3).cpu().numpy() - global_trans
            return {
                "right": {
                    "v3d": right_v,
                    "f3d": self.layers["right"].faces,
                    "vc": None,
                    "name": f"{seq_name}_right",
                    "color": "white",
                }
            }
        elif 'rot_l' in batch and 'pose_l' in batch:
            nf = batch["rot_l"].shape[0]
            shape_l = batch["shape_l"]
            if shape_l.ndim == 1:
                shape_l = shape_l.unsqueeze(0).repeat(nf, 1)
            left_v = self.layers["left"](
                global_orient=batch['rot_l'].to(torch.float32),
                hand_pose=batch['pose_l'].to(torch.float32),
                betas=shape_l.to(torch.float32),
            )
            # NOTE Use this to solve the misalignment between MANO and smplx
            left_v = left_v.vertices.detach().cpu().numpy() - left_v.joints[:, 0:1, :].detach().cpu().numpy()
            # left_v = left_v - left_v[:, 0:1]
            left_v = left_v + batch["trans_l"].view(nf, 1, 3).cpu().numpy() - global_trans
            return {
                "left": {
                    "v3d": left_v,
                    "f3d": self.layers["left"].faces,
                    "vc": None,
                    "name": f"{seq_name}_left",
                    "color": "white",
                }
            }
        else:
            return {}

    def _sanity_check(self, seq):
        assert 'rot_r' in seq

    def format_gt(self, seq):
        if "keypoint" in seq:
            # import ipdb; ipdb.set_trace()
            seq.update({
                "pos_l": seq['keypoint'][:, :21],
                "pos_r": seq['keypoint'][:, 21:],
                "rot_l": seq["pose"][:, :3],
                "pose_l": seq["pose"][:, 3:48],
                "rot_r": seq["pose"][:, 48:51],
                "pose_r": seq["pose"][:, 51:],
                "trans_l": seq['keypoint'][:, 0, :],
                "trans_r": seq['keypoint'][:, 21, :],
                "trans_obj": seq['obj_pose'][:, :3],
                "rot_obj": seq['obj_pose'][:, 3:],
                "arti_obj": seq['obj_arti'] if 'arti_obj' in seq else None,
                "type_obj": seq['obj_type']
            })
        return seq

    def get_hand_mesh(self, batch, seq_name='', glob_viewer_rot=None, glob_viewer_trans=None):
        all_meshes = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(self.dev)
        print(batch.keys())
        glob_viewer_trans = 0 if glob_viewer_trans is None else glob_viewer_trans

        if self.dataset == 'arctic':
            all_meshes.update(self.arctic_obj_mesh(batch, seq_name, glob_viewer_trans))
        elif self.dataset == 'grab':
            all_meshes.update(self.grab_obj_mesh(batch, seq_name, glob_viewer_trans))
        all_meshes.update(self.arctic_hand_mesh(batch, seq_name, glob_viewer_trans, is_right=True))
        all_meshes.update(self.arctic_hand_mesh(batch, seq_name, glob_viewer_trans, is_right=False))
            # NOTE Do this to flip right to left
            # fake_pose_r = (batch["pose_l"] + self.layers["left"].hand_mean).view(-1, 15, 3)
            # print(fake_pose_r.norm(dim=-1))
            # fake_pose_r = normalize_rot_vec_sequence(fake_pose_r)
            # fake_pose_r[...,1:] = -1 * fake_pose_r[...,1:]
            # fake_pose_r = fake_pose_r.view(-1, 45)
            # fake_rot_r = batch["rot_l"]
            # fake_rot_r[...,1:] = -1 * fake_rot_r[...,1:]

        meshes = construct_viewer_meshes(
                all_meshes,
                draw_edges=False,
                flat_shading=False,
                glob_rot=glob_viewer_rot
        )
        data = ViewerData(num_frames=batch['rot_r'].shape[0])
        return meshes, data

    def render_seq(self, seqs, vid_p=".exps/debug/render.mp4", recenter=True):
        self.v.reset()
        for seq_name, seq in seqs.items():
            for k in seq:
                if isinstance(seq[k], torch.Tensor):
                    seq[k] = seq[k].cpu().numpy()
            if seq_name == 'gt':
                seq = self.format_gt(seq)
            self._sanity_check(seq)
            if recenter:
                glob_trans = seq['trans_obj'][0].reshape(1, 1, 3)
                # flip_mat1 = aa2rot_numpy(np.array([0, -1, 0]) * np.pi/2)
                flip_mat = aa2rot_numpy(np.array([-1, 0, 0]) * np.pi/2)
                # flip_mat = np.matmul(flip_mat1, flip_mat2)
                # init_obj_rot_mat = aa2rot_numpy(seq['rot_obj'][0])
                glob_rot_mat = flip_mat #np.matmul(flip_mat, init_obj_rot_mat.transpose(0, 1))
            else:
                glob_trans = None
                glob_rot = None
            skeletons = self.get_hand_skeleton(seq, seq_name, glob_rot_mat, glob_trans)
            meshes, data = self.get_hand_mesh(seq, seq_name, glob_rot_mat, glob_trans)
            for skeleton in skeletons:
                self.v.scene.add(skeleton)
            for mesh in meshes.values():
                self.v.scene.add(mesh)

        if not self.gui_initialized:
            self.setup_viewer()
            self.gui_initialized = True

            if self.interactive:
                self.view_interactive()
            else:
                self.view_fn_headless(vid_p)
        else:
            self.v.scene.ctx = self.v.ctx
            self.v.scene.origin.enabled = False
            self.v.scene.floor.enabled = False
            if not self.interactive:
                self.view_fn_headless(vid_p)
            # self.v.auto_set_floor = False
            # self.v.scene.floor.position[1] = -3
            # self.v.scene.camera.position = np.array((0.0, -1.0, 0))