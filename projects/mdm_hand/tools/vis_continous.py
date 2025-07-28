import aitviewer
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.meshes import Meshes
from aitviewer.headless import HeadlessRenderer
from aitviewer.viewer import Viewer
import numpy as np
import os
import torch
import numpy as np
import argparse
import trimesh
import scipy
import colorsys

VERTS_OPEN = [
    [78, 121],
    [121, 214],
    [214, 215],
    [215, 279],
    [279, 239],
    [239, 234],
    [234, 92],
    [92, 38],
    [38, 122],
    [122, 118],
    [118, 117],
    [117, 119],
    [119, 120],
    [120, 108],
    [108, 79],
    [79, 78],
]

verts_open_id = list(set([v for edge in VERTS_OPEN for v in edge]))

A_MIN = 0.3
DA = 0.9


def make_mano_sequence_wt(verts_hand_l, verts_hand_r, faces_hand_l, faces_hand_r):

    # compute barycenters from right hand loops
    barycenter_r = np.mean(verts_hand_r[:, verts_open_id, :], axis=1)[:, np.newaxis, :]
    # add as new vert
    verts_hand_r_wt = np.concatenate([verts_hand_r, barycenter_r], 1)
    extra_vert_ID = verts_hand_r_wt.shape[1] - 1
    # add new faces
    extra_faces = np.array([edge + [extra_vert_ID] for edge in VERTS_OPEN])
    faces_hand_r_wt = np.concatenate([faces_hand_r, extra_faces])

    # compute barycenters from right hand loops
    barycenter_l = np.mean(verts_hand_l[:, verts_open_id, :], axis=1)[:, np.newaxis, :]
    # add as new vert
    verts_hand_l_wt = np.concatenate([verts_hand_l, barycenter_l], 1)
    extra_vert_ID = verts_hand_l_wt.shape[1] - 1
    # add new faces
    extra_faces = np.array([edge[::-1] + [extra_vert_ID] for edge in VERTS_OPEN])
    faces_hand_l_wt = np.concatenate([faces_hand_l, extra_faces])

    return verts_hand_l_wt, verts_hand_r_wt, faces_hand_l_wt, faces_hand_r_wt


RGB_L = (121 / 255.0, 119 / 255.0, 158 / 255.0)
RGB_R = (158 / 255.0, 121 / 255.0, 119 / 255.0)
RGB_O = (121 / 255.0, 140 / 255.0, 119 / 255.0)

HSV_L = colorsys.rgb_to_hsv(*RGB_L)
HSV_R = colorsys.rgb_to_hsv(*RGB_R)
HSV_O = colorsys.rgb_to_hsv(*RGB_O)

A_MIN = 0.3
DA = 0.8


def render_sequence(pose_data_path, set_object_on_floor):

    files = os.listdir(os.path.join(pose_data_path))

    for fn in files:
        mesh_dict = np.load(os.path.join(pose_data_path, fn), allow_pickle=True).item()
        verts_hand_l = mesh_dict["verts_hand_l"]
        verts_hand_r = mesh_dict["verts_hand_r"]
        verts_obj = mesh_dict["verts_obj"]

        faces_hand_l = mesh_dict["faces_hand_l"]
        faces_hand_r = mesh_dict["faces_hand_r"]
        faces_obj = mesh_dict["obj_faces"]

        verts_hand_l, verts_hand_r, faces_hand_l, faces_hand_r = make_mano_sequence_wt(
            verts_hand_l, verts_hand_r, faces_hand_l, faces_hand_r
        )

        # move to floor
        if set_object_on_floor:
            # compute minimum z at frame 0 for obj
            min_y_obj = np.min(verts_obj[0, :, 2])
            verts_obj[:, :, 2] -= min_y_obj
            verts_hand_r[:, :, 2] -= min_y_obj
            verts_hand_l[:, :, 2] -= min_y_obj

        # SETUP rendering
        v = HeadlessRenderer(size=(4000, 4000))
        v.scene.camera.fov = 40
        # target should be the center of the object at frame 0
        # set target to mid point of the object motion
        v.scene.camera.target = [
            0.5 * (np.min(verts_obj, axis=(0, 1)) + np.max(verts_obj, axis=(0, 1)))
        ]
        v.scene.camera.position = [0.0, 0.5, 2.0]
        v.scene.camera.up = [0, 1.0, 0]
        v.auto_set_camera_target = False
        # v.scene.remove(v.scene.lights[0])
        v.scene.remove(v.scene.origin)
        # v.scene.remove(v.scene.floor)

        # select key_frames
        FRAME_IDS = [14, 35, 59, 100]

        hand_l_mesh = []
        hand_r_mesh = []
        obj_mesh = []

        for idx, frame_id in enumerate(FRAME_IDS):
            id_verts_hand_l = verts_hand_l[frame_id][np.newaxis, ...]
            id_verts_hand_r = verts_hand_r[frame_id][np.newaxis, ...]
            id_verts_obj = verts_obj[frame_id][np.newaxis, ...]
            hand_l_mesh = Meshes(id_verts_hand_l, faces_hand_l, z_up=True)
            hand_r_mesh = Meshes(id_verts_hand_r, faces_hand_r, z_up=True)
            obj_mesh = Meshes(id_verts_obj, faces_obj, z_up=True)
            # SET COLOR
            alpha = A_MIN + DA * (float(idx) / (len(FRAME_IDS) - 1))
            rgb_L = colorsys.hsv_to_rgb(HSV_L[0], alpha * HSV_L[1], HSV_L[2])
            rgb_R = colorsys.hsv_to_rgb(HSV_R[0], alpha * HSV_R[1], HSV_R[2])
            rgb_O = colorsys.hsv_to_rgb(HSV_O[0], alpha * HSV_O[1], HSV_O[2])

            hand_l_mesh.material.color = (
                rgb_L[0],
                rgb_L[1],
                rgb_L[2],
                1.0,
            )  # 0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            hand_r_mesh.material.color = (
                rgb_R[0],
                rgb_R[1],
                rgb_R[2],
                1.0,
            )  # 0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            obj_mesh.material.color = (
                rgb_O[0],
                rgb_O[1],
                rgb_O[2],
                1.0,
            )  # 0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            v.scene.add(obj_mesh, hand_l_mesh, hand_r_mesh)

        # Now add trajectories

        def interpolate(sequence, N=100):
            sequences = []
            for i in range(N):
                alpha_interpolate = float(i) / (N - 1)

                sequences.append(
                    alpha_interpolate * sequence[0, 1:, :]
                    + (1.0 - alpha_interpolate) * sequence[0, 0:-1, :]
                )

            return np.stack(sequences, axis=0).transpose(1, 0, 2).reshape(1, -1, 3)

        if False:
            wrist_l = interpolate(
                np.mean(
                    verts_hand_l[FRAME_IDS[0] : FRAME_IDS[-1], verts_open_id, :], axis=1
                )[np.newaxis, ...]
            )
            wrist_r = interpolate(
                np.mean(
                    verts_hand_r[FRAME_IDS[0] : FRAME_IDS[-1], verts_open_id, :], axis=1
                )[np.newaxis, ...]
            )
            obj = interpolate(
                np.mean(verts_obj[FRAME_IDS[0] : FRAME_IDS[-1], :, :], axis=1)[
                    np.newaxis, ...
                ]
            )

            sphere_l = Spheres(wrist_l)
            sphere_l.color = (
                200 / 255.0,
                200 / 255.0,
                200 / 255.0,
                1.0,
            )  # 0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            sphere_l.rotation = np.matmul(
                np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), sphere_l.rotation
            )
            sphere_l.radius *= 0.2

            sphere_r = Spheres(wrist_r)
            sphere_r.color = (
                200 / 255.0,
                200 / 255.0,
                200 / 255.0,
                1.0,
            )  # 0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            sphere_r.rotation = np.matmul(
                np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), sphere_r.rotation
            )
            sphere_r.radius *= 0.2

            sphere_obj = Spheres(obj)
            sphere_obj.color = (
                200 / 255.0,
                200 / 255.0,
                200 / 255.0,
                1.0,
            )  # 0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            sphere_obj.rotation = np.matmul(
                np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), sphere_obj.rotation
            )
            sphere_obj.radius *= 0.2

            v.scene.add(sphere_r, sphere_l, sphere_obj)

        """
        for i in range(wrist_l.shape[1]):


            alpha_interpolate = 3*A_MIN + DA*float(i)/(wrist_l.shape[1]-1)
            print(alpha_interpolate)

            sphere_l = Spheres(wrist_l[:,i,][np.newaxis,:])
            sphere_l.color = (alpha_interpolate*121/255.0, alpha_interpolate*119/255.0, alpha_interpolate*158/255.0,1.0) #0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            sphere_l.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), sphere_l.rotation)
            sphere_l.radius *= 0.3

            sphere_r = Spheres(wrist_r[:,i,][np.newaxis,:])
            sphere_r.color = (alpha_interpolate*158/255.0, alpha_interpolate*121/255.0, alpha_interpolate*119/255.0, 1.0) #0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            sphere_r.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), sphere_r.rotation)
            sphere_r.radius *= 0.3
            sphere_obj = Spheres(obj[:,i,][np.newaxis,:])
            sphere_obj.color = (alpha_interpolate*121/255.0, alpha_interpolate*140/255.0, alpha_interpolate*119/255.0,1.0) #0.3+0.7*(float(idx)/(len(FRAME_IDS)-1)))
            sphere_obj.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), sphere_obj.rotation)
            sphere_obj.radius *= 0.3

            v.scene.add(sphere_r, sphere_l, sphere_obj)
        """
        # v.run()

        v.save_video("method/" + fn, quality="high")
        v.scene.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with command line arguments")

    parser.add_argument(
        "--file_path", type=str, required=True, help="The file path argument"
    )

    parser.add_argument(
        "--set_object_on_floor",
        action="store_true",
        default=False,
        help="Weather to set object on ground",
    )

    args = parser.parse_args()
    render_sequence(args.file_path, args.set_object_on_floor)