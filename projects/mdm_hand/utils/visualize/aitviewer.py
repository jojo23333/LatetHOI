import os
import os.path as op
import re
from abc import abstractmethod

import matplotlib.cm as cm
import numpy as np
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material
from easydict import EasyDict as edict
from loguru import logger
from PIL import Image
from aitviewer.utils.so3 import aa2rot_numpy

OBJ_ID = 100
SMPLX_ID = 150
LEFT_ID = 200
RIGHT_ID = 250
SEGM_IDS = {"object": OBJ_ID, "smplx": SMPLX_ID, "left": LEFT_ID, "right": RIGHT_ID}
MANO_HAND_SKELETON = [
    [-1, 0],
    [0, 1], [1, 2], [2, 3], [3, 17],
    [0, 4], [4, 5], [5, 6], [6, 18],
    [0, 7], [7, 8], [8, 9], [9, 20],
    [0, 10], [10, 11], [11, 12], [12, 19],
    [0, 13], [13, 14], [14, 15], [15, 16],
]

cmap = cm.get_cmap("plasma")
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
}

class ViewerData(edict):
    """
    Interface to standardize viewer data.
    """

    def __init__(self, Rt=None, K=None, num_frames=1 ):
        self.Rt = Rt
        self.K = K
        self.num_frames = num_frames
        self.validate_format()

    def validate_format(self):
        pass

def dist2vc(dist_ro, dist_lo, dist_o, _cmap, tf_fn=None):
    if tf_fn is not None:
        exp_map = tf_fn
    else:
        exp_map = small_exp_map
    dist_ro = exp_map(dist_ro)
    dist_lo = exp_map(dist_lo)
    dist_o = exp_map(dist_o)

    vc_ro = _cmap(dist_ro)
    vc_lo = _cmap(dist_lo)
    vc_o = _cmap(dist_o)
    return vc_ro, vc_lo, vc_o


def small_exp_map(_dist):
    dist = np.copy(_dist)
    # dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    dist = np.exp(-20.0 * dist)
    return dist

def construct_hand_skeleton(joints, color, name='Skeleton', glob_rot=None, glob_trans=None):
    return Skeletons(
        joints,
        MANO_HAND_SKELETON,
        gui_affine=False,
        color=materials[color]._color,
        radius=0.005,
        name=name,
        position=glob_trans,
        rotation=glob_rot
    )

def construct_viewer_meshes(data, draw_edges=False, flat_shading=True, glob_rot=None, glob_trans=None):
    # rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
    meshes = {}
    for key, val in data.items():
        if "object" in key:
            flat_shading = False
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
            material=materials[val["color"]],
            position=glob_trans,
            rotation=glob_rot,
            # rotation=rotation_flip,
        )
    return meshes


def setup_viewer(
    v, shared_folder_p, video, images_path, data, flag, seq_name, side_angle
):
    fps = 10
    cols, rows = 224, 224
    focal = 1000.0

    # setup image paths
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    # setup billboard
    images_path = op.join(shared_folder_p, "images")
    images_paths = [
        os.path.join(images_path, f)
        for f in sorted(os.listdir(images_path), key=sort_key)
    ]
    assert len(images_paths) > 0

    cam_t = data[f"{flag}.object.cam_t"]
    num_frames = min(cam_t.shape[0], len(images_paths))
    cam_t = cam_t[:num_frames]
    # setup camera
    K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :, 3] = cam_t
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if side_angle is None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    v.scene.add(camera)
    v.run_animations = True  # autoplay
    v.playback_fps = fps
    v.scene.fps = fps
    v.scene.origin.enabled = False
    v.scene.floor.enabled = False
    v.auto_set_floor = False
    v.scene.floor.position[1] = -8
    v.set_temp_camera(camera)
    # v.scene.camera.position = np.array((0.0, 0.0, 0))
    return v


def render_depth(v, depth_p):
    depth = np.array(v.get_depth()).astype(np.float16)
    np.save(depth_p, depth)


def render_mask(v, mask_p):
    nodes_uid = {node.name: node.uid for node in v.scene.collect_nodes()}
    my_cmap = {
        uid: [SEGM_IDS[name], SEGM_IDS[name], SEGM_IDS[name]]
        for name, uid in nodes_uid.items()
        if name in SEGM_IDS.keys()
    }
    mask = np.array(v.get_mask(color_map=my_cmap)).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(mask_p)


def setup_billboard(data, v):
    images_paths = data.imgnames
    K = data.K
    Rt = data.Rt
    rows = data.rows
    cols = data.cols
    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if images_paths is not None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    v.scene.add(camera)
    v.scene.camera.load_cam()
    v.set_temp_camera(camera)
