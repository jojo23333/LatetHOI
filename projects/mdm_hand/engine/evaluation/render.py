
import torch
import numpy as np
import torch
import os
import logging

from aitviewer.renderables.meshes import Meshes
from aitviewer.headless import HeadlessRenderer
from utils.visualize.viewer import Viewer
from aitviewer.scene.material import Material
from aitviewer.configuration import CONFIG as C
C.window_type = "pyglet"
C.export_path = './'
# C.z_up = True

logger = logging.getLogger("d2")
materials = {
    "none": None,
    "white": Material(color=(1.0, 1.0, 1.0, 1.0), ambient=0.2),
    "red": Material(color=(0.969, 0.106, 0.059, 1.0), ambient=0.2),
    "blue": Material(color=(0.0, 0.0, 1.0, 1.0), ambient=0.2),
    "green": Material(color=(1.0, 0.0, 0.0, 1.0), ambient=0.2),
    "cyan": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "light-blue": Material(color=(0.588, 0.5647, 0.9725, 1.0), ambient=0.2),
    "cyan-light": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "dark-light": Material(color=(0.404, 0.278, 0.278, 1.0), ambient=0.2),
    "rice": Material(color=(0.922, 0.922, 0.102, 1.0), ambient=0.2),
    'color_left': Material(color=(1.0, 218/255, 190/255, 1.0), ambient=0.2),
    'color_right': Material(color=(240/255, 184/255, 160/255, 1.0), ambient=0.2),
}

class MotionRender:
    def __init__(
        self,
        smoothing=False,
        dexycb=False,
        render_types=["rgb", "depth", "mask"],
        interactive=True,
        size=(1518, 2024)
    ):
        self.smoothing = smoothing
        self.dexycb = dexycb
        self.save_only = False
        if not interactive:
            try:
                v = HeadlessRenderer(size=size, backend='egl')
            except Exception as e:
                logger.warning(e)
                logger.warning("Only save target .npy")
                self.save_only = True
                v = None
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
        v.scene.fps = 20
        v.playback_fps = 20
        v._init_scene()
        if not os.path.exists(os.path.dirname(vid_p)):
            os.makedirs(os.path.dirname(vid_p))

        logger.info("Rendering to video" + vid_p)
        v.save_video(video_dir=vid_p, output_fps=20, duration=8)

    def setup_viewer(self):
        v = self.v
        fps = 20
        # camera.show_path()
        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False

        # cam_dict = joblib.load(cam_dir + "cam_params.pkl")
        if self.dexycb:
            # cam_dict = {
            #     'position': np.array([ 0.03641457, -0.91735137,  0.4832493 ], dtype=np.float32),
            #     'target': np.array([ 0.0374573 , -0.04785523,  0.01903416], dtype=np.float32),
            #     'up': np.array([0., 0., 1.], dtype=np.float32),
            #     'ZOOM_FACTOR': 8,
            #     'ROT_FACTOR': 0.0025,
            #     'PAN_FACTOR': 0.01,
            #     'near': 0.01,
            #     'far': 1000.0
            # }
            cam_dict = {
                'position': np.array([-0.27407348, -0.73361474,  0.5323694 ], dtype=np.float32),
                'target': np.array([ 0.00601383, -0.02391038,  0.08263678], dtype=np.float32),
                'up': np.array([0., 0., 1.], dtype=np.float32),
                'ZOOM_FACTOR': 8,
                'ROT_FACTOR': 0.0025,
                'PAN_FACTOR': 0.01,
                'near': 0.01,
                'far': 1000.0
            }
        else:
            cam_dict =  {
                'position': np.array([ 0.38526526,  0.5168136 , -1.4761424 ], dtype=np.float32),
                'target': np.array([-0.08916569,  0.05183123,  0.03438002], dtype=np.float32),
                'up': np.array([0., 1., 0.], dtype=np.float32),
                'ZOOM_FACTOR': 8,
                'ROT_FACTOR': 0.0025,
                'PAN_FACTOR': 0.01,
                'near': 0.01,
                'far': 1000.0
            }

        v.scene.camera.position = cam_dict["position"]
        v.scene.camera.target = cam_dict["target"]
        v.scene.camera.up = cam_dict["up"]
        v.scene.camera.ZOOM_FACTOR = cam_dict["ZOOM_FACTOR"]
        v.scene.camera.ROT_FACTOR = cam_dict["ROT_FACTOR"]
        v.scene.camera.PAN_FACTOR = cam_dict["PAN_FACTOR"]
        v.scene.camera.near = cam_dict["near"]
        v.scene.camera.far = cam_dict["far"]
        self.v = v

    def construct_viewer_meshes(self, data, draw_edges=False, flat_shading=True):
    # rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
        meshes = {}
        for key, val in data.items():
            if "obj" in key:
                flat_shading = False
                v3d = val["v3d"]
                if hasattr(self, 'obj_vertex_colors'):
                    meshes[key] = Meshes(
                        v3d,
                        val["f3d"],
                        face_colors=self.obj_face_colors,
                        vertex_colors=self.obj_vertex_colors,
                        uv_coords=self.obj_uvs,
                        path_to_texture=self.obj_texture_path,
                        name=val["name"],
                        flat_shading=flat_shading,
                        draw_edges=draw_edges
                    )
                else:
                    meshes[key] = Meshes(
                        v3d,
                        val["f3d"],
                        vertex_colors=val["vc"],
                        name=val["name"],
                        flat_shading=flat_shading,
                        draw_edges=draw_edges,
                        material=materials[val["color"]]
                    )
            else:
                flat_shading = flat_shading
                v3d = val["v3d"]
                meshes[key] = Meshes(
                    v3d,
                    val["f3d"],
                    vertex_colors=val["vc"],
                    name=val["name"],
                    flat_shading=flat_shading,
                    draw_edges=draw_edges,
                    material=materials[val["color"]]
                )
        return meshes
    
    def temporal_smoothing(self, o_v, rh_v, lh_v):
        pov = np.pad(o_v, ((1, 1), (0, 0), (0, 0)), mode='edge')
        o_v = pov[:-2, :, :]*0.25 + pov[1:-1,:,:]*0.5 + pov[2:,:,:]*0.25
        prhv = np.pad(rh_v, ((1, 1), (0, 0), (0, 0)), mode='edge')
        rh_v = prhv[:-2, :, :]*0.25 + prhv[1:-1,:,:]*0.5 + prhv[2:,:,:]*0.25
        if lh_v is not None:
            plhv = np.pad(lh_v, ((1, 1), (0, 0), (0, 0)), mode='edge')
            lh_v = plhv[:-2, :, :]*0.25 + plhv[1:-1,:,:]*0.5 + plhv[2:,:,:]*0.25
        return o_v, rh_v, lh_v

    def get_mesh(self, o_v, o_f, rh_v, rh_f, lh_v=None, lh_f=None,  seq_name=''):
        if isinstance(o_v, torch.Tensor):
            o_v = o_v.cpu().detach().numpy()
        if isinstance(o_f, torch.Tensor):
            o_f = o_f.cpu().detach().numpy()
        if isinstance(rh_v, torch.Tensor):
            rh_v = rh_v.cpu().detach().numpy()
        if isinstance(rh_f, torch.Tensor):
            rh_f = rh_f.cpu().cpu().detach().numpy()
        if lh_v is not None and isinstance(lh_v, torch.Tensor):
            lh_v = lh_v.cpu().detach().numpy()
        if lh_f is not None and isinstance(lh_f, torch.Tensor):
            lh_f = lh_f.cpu().detach().numpy()
        if self.smoothing:
            o_v, rh_v, lh_v = self.temporal_smoothing(o_v, rh_v, lh_v)

        all_meshes = {}
        def fix_xyz(v):
            v = v[:, :, [0, 2, 1]]
            v[:, :, 2] = -v[:, :, 2]
            return v
        # def fix_xyz(v):
        #     return v
        
        # print((o_v[40] - o_v[0]).mean(axis=-2))
        all_meshes["obj"] = {
            "v3d": fix_xyz(o_v),
            "f3d": o_f,
            "vc": None,
            "name": f"{seq_name}_object",
            "color": "cyan-light" if seq_name =='gt' else "light-blue",
        }
        all_meshes["right"] = {
            "v3d": fix_xyz(rh_v),
            "f3d": rh_f,
            "vc": None,
            "name": f"{seq_name}_right",
            "color": "color_right",
        }
        if lh_v is not None:
            all_meshes["left"] = {
                "v3d": fix_xyz(lh_v),
                "f3d": lh_f,
                "vc": None,
                "name": f"{seq_name}_left",
                "color": "color_left",
            }
        meshes = self.construct_viewer_meshes(
                all_meshes,
                draw_edges=False,
                flat_shading=False,
        )
        return meshes
    
    def open_file(self):
        try:
            from PyQt6.QtWidgets import QApplication, QFileDialog
        except ImportError:
            from PyQt5.QtWidgets import QApplication, QFileDialog
            print("PyQt6 not found, imported PyQt5 instead.")
        dialog = QFileDialog()
        file_name, _ = dialog.getOpenFileName(caption='Select file', directory='/Volumes/graphics/nimble/users/muchenli/EXP_OUTPUT/hdm')
        if file_name.endswith('npy'):
            data = np.load(file_name, allow_pickle=True).item()
        else:
            data = torch.load(file_name, map_location='cpu')
        print("Opening: ", file_name)
        self.render_seq(data)

    def render_seq(self, seqs, vid_p=".exps/debug/render.mp4", save_pth=True, obj_mesh=None):
        if save_pth:
            export_path = vid_p.replace('.mp4', '.pth')
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            torch.save(seqs, export_path)
        
        if obj_mesh is not None:
            import trimesh
            mesh = trimesh.load(obj_mesh)
            self.obj_uvs = None
            self.obj_vertex_colors = None
            self.obj_face_colors = None
            if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                if mesh.visual.kind == "vertex_colors":
                    self.obj_vertex_colors = mesh.visual.vertex_colors
                elif mesh.visual.kind == "face_colors":
                    self.obj_face_colors = mesh.visual.vertex_colors
            elif isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                self.obj_uvs = mesh.visual.uv
            obj_dir = os.path.dirname(obj_mesh)
            material_file = [x for x in os.listdir(obj_dir) if x.startswith('material_0')][0]
            self.obj_texture_path = os.path.join(obj_dir, material_file)

        if self.save_only:
            return

        self.v.reset()
        for seq_name, seq in seqs.items():
            o_v, o_f, rh_v, rh_f, lh_v, lh_f = seq
            meshes = self.get_mesh(
                o_v, o_f, rh_v, rh_f, lh_v, lh_f, seq_name
            )
            for mesh in meshes.values():
                self.v.scene.add(mesh)

        self.setup_viewer()

        if self.interactive:
            self.view_interactive()
        else:
            self.view_fn_headless(vid_p)
        # else:
        #     self.v.scene.ctx = self.v.ctx
        #     self.v.scene.origin.enabled = False
        #     self.v.scene.floor.enabled = False
        #     if not self.interactive:
        #         self.view_fn_headless(vid_p)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script with command line arguments")
    parser.add_argument(
        '-f', "--file_path", type=str, required=True, help="The file path argument"
    )
    parser.add_argument("--smooth", action='store_true')
    parser.add_argument("--dex", action='store_true')
    parser.add_argument("--obj", action='store_true')
    args = parser.parse_args()
    MOTION_RENDER = MotionRender(smoothing=args.smooth, dexycb=args.dex, interactive=False)
    assert os.path.exists(args.file_path), "File do not exist"
    assert args.file_path.endswith('.pth'), "Wrong format"
    export_vid_path = args.file_path.replace('.pth', '.mp4')
    seq = torch.load(args.file_path, map_location='cpu')
    if isinstance(seq, dict):
        seq = seq['pred']
    if args.obj:
        obj_name = '_'.join(os.path.basename(args.file_path).split('_')[1:3])
        obj_mesh = os.path.join('data/oakink/selected_texture_uv', obj_name, obj_name+'.obj')
    else:
        obj_mesh = None
    
    MOTION_RENDER.render_seq({os.path.basename(args.file_path): seq}, vid_p=export_vid_path, save_pth=False, obj_mesh=obj_mesh)
    
    
    