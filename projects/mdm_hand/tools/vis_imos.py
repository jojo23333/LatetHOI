import aitviewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.scene.camera import PinholeCamera, ViewerCamera
from aitviewer.viewer import Viewer
import os
import pickle
import csv
import numpy as np
import smplx.joint_names
import torch
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import trimesh
from manopth.manolayer import ManoLayer
from aitviewer.configuration import CONFIG as C
from tqdm import tqdm
C.smplx_models = 'data/body_models'


def compute_global_orientations(local_rotations, parents):
    global_rotations = np.tile(
        np.eye(3), (len(local_rotations), local_rotations.shape[1], 1, 1)
    )
    global_rotations[:, 0] = local_rotations[:, 0]
    for i in range(1, len(local_rotations)):
        global_rotations[:, i] = np.matmul(
            global_rotations[:, parents[i]], local_rotations[:, i]
        )
    return global_rotations


if __name__ == "__main__":
    in_dir = "/data/muchenli/MLCode/projects/mdm_hand/.exps/SELECTED_RESULTS/IMOS"

    v = HeadlessRenderer(size=(4000, 3000))
    folders = os.listdir(in_dir)
    folder = 'test'
    files = os.listdir(os.path.join(in_dir, folder))
    export_dir = os.path.join(in_dir, 'out')
    hand_r_mesh_color = (121 / 255.0, 119 / 255.0, 158 / 255.0, 1.0)
    hand_l_mesh_color = (158 / 255.0, 121 / 255.0, 119 / 255.0, 1.0)
    obj_mesh_color = (121 / 255.0, 140 / 255.0, 119 / 255.0, 1.0)

    for fn in tqdm(files):
        fp = os.path.join(in_dir, folder, fn)
        seq_dict = torch.load(fp, map_location=torch.device("cpu"))

        sbj_id = fn.split("_")[0]
        obj_name = fn.split("_")[1]
        intent_name = fn.split("_")[2]
        rest = ''.join(fn.split("_")[3:]).replace('.npz', '.pth')
        export_name = f'{sbj_id}_{obj_name}_{intent_name}_{rest}'

        obj_pose = seq_dict["obj_p"]

        switch_point = (
            np.linalg.norm(
                obj_pose["transl"].detach().numpy()[:-1]
                - obj_pose["transl"].detach().numpy()[1:],
                axis=-1,
            ).argmax()
            + 1
        )
        obj_trans = (
            obj_pose["transl"].detach().numpy()
            - obj_pose["transl"].detach().numpy()[:1]
        )
        obj_orient_aa = obj_pose["global_orient"]
        obj_orient_quat = R.from_rotvec(obj_orient_aa).as_quat()
        obj_orient = R.from_rotvec(obj_orient_aa).as_matrix()
        obj_mesh = trimesh.load("data/grab/tools/object_meshes/contact_meshes/" + obj_name + ".ply")

        T = obj_trans.shape[0]
        T1 = 5 * T

        trans_x_interp = np.zeros((T1, 1))
        trans_y_interp = np.zeros((T1, 1))
        trans_z_interp = np.zeros((T1, 1))
        rot_x_interp = np.zeros((T1, 1))
        rot_y_interp = np.zeros((T1, 1))
        rot_z_interp = np.zeros((T1, 1))
        rot_w_interp = np.zeros((T1, 1))

        x = np.linspace(0, T - 1, T)
        x_new = np.linspace(0, T - 1, T1)

        # import pdb; pdb.set_trace()
        slerp = Slerp(x, R.from_rotvec(obj_orient_aa))
        slerp_mat = slerp(x_new).as_matrix()
        # for v1 in range(0, obj_verts.shape[1]):
        trans_obj_x = obj_trans[:, 0]
        trans_obj_y = obj_trans[:, 1]
        trans_obj_z = obj_trans[:, 2]
        rot_obj_x = obj_orient_quat[:, 0]
        rot_obj_y = obj_orient_quat[:, 1]
        rot_obj_z = obj_orient_quat[:, 2]
        rot_obj_w = obj_orient_quat[:, 3]
        f_x = interpolate.interp1d(x, trans_obj_x, kind="linear")
        f_y = interpolate.interp1d(x, trans_obj_y, kind="linear")
        f_z = interpolate.interp1d(x, trans_obj_z, kind="linear")
        f_x_rot = interpolate.interp1d(x, rot_obj_x, kind="linear")
        f_y_rot = interpolate.interp1d(x, rot_obj_y, kind="linear")
        f_z_rot = interpolate.interp1d(x, rot_obj_z, kind="linear")
        f_w_rot = interpolate.interp1d(x, rot_obj_w, kind="linear")
        trans_x_interp = f_x(x_new)
        trans_y_interp = f_y(x_new)
        trans_z_interp = f_z(x_new)
        rot_x_interp = f_x_rot(x_new)
        rot_y_interp = f_y_rot(x_new)
        rot_z_interp = f_z_rot(x_new)
        rot_w_interp = f_w_rot(x_new)

        trans_x_interp = torch.from_numpy(trans_x_interp).unsqueeze(1)
        trans_y_interp = torch.from_numpy(trans_y_interp).unsqueeze(1)
        trans_z_interp = torch.from_numpy(trans_z_interp).unsqueeze(1)
        rot_x_interp = torch.from_numpy(rot_x_interp).unsqueeze(1)
        rot_y_interp = torch.from_numpy(rot_y_interp).unsqueeze(1)
        rot_z_interp = torch.from_numpy(rot_z_interp).unsqueeze(1)
        rot_w_interp = torch.from_numpy(rot_w_interp).unsqueeze(1)
        trans_interp = torch.cat(
            (trans_x_interp, trans_y_interp, trans_z_interp), dim=-1
        )
        rot_interp = torch.cat(
            (rot_x_interp, rot_y_interp, rot_z_interp, rot_w_interp), dim=-1
        )
        trans_all = torch.cat(
            (torch.from_numpy(obj_trans[:1]), trans_interp), dim=0
        ).numpy()

        rot_all = torch.cat(
            (torch.tensor(obj_orient_quat[:1]), rot_interp), dim=0
        ).numpy()

        rot_all_mat = R.from_quat(rot_all).as_matrix()
        # switch_point *= (T)
        ##verts_all[switch_point-T:switch_point] = verts_all[switch_point-T:switch_point-T+1]

        obj_verts = np.matmul(obj_mesh.vertices, slerp_mat)
        obj_verts += trans_all[:-1, np.newaxis]

        vertex_colors = np.tile(
            [0.5, 0.1, 0.5, 1], (obj_verts.shape[0], obj_verts.shape[1], 1)
        )
        vertex_colors[:, : obj_verts.shape[1] // 2, :] = [
            158 / 255.0,
            121 / 255.0,
            119 / 255.0,
            1.0,
        ]

        mesh_frame = Meshes(
            obj_verts, obj_mesh.faces, color=obj_mesh_color
        )

        sbj_pose = seq_dict["sbj_p"]
        sbj_trans = (
            sbj_pose["transl"].detach().numpy()
            - obj_pose["transl"].detach().numpy()[:1]
        )
        sbj_orient = R.from_matrix(
            sbj_pose["global_orient"].detach().numpy().reshape(-1, 3, 3)
        ).as_rotvec()

        if seq_dict["gender"][0] == 2:
            gender = "female"
        else:
            gender = "male"

        sbj_vtemp_lhand = trimesh.load(
            os.path.join(
                "data/grab/tools/subject_meshes", gender, sbj_id + "_lhand.ply"
            )
        )
        sbj_vtemp_rhand = trimesh.load(
            os.path.join(
                "data/grab/tools/subject_meshes", gender, sbj_id + "_rhand.ply"
            )
        )
        # TODO
        hand_idxs_left = np.load('data/grab/tools/smplx_correspondence/lhand_smplx_ids.npy')
        hand_idxs_right= np.load('data/grab/tools/smplx_correspondence/rhand_smplx_ids.npy')

        smpl_layer = SMPLLayer(
            model_type="smplx",
            gender=gender,
            v_template=seq_dict["sbj_vtemp"],
            use_pca=False,
            num_pca_comps=48,
            is_rhand=True,
            flat_hand_mean=True,
        )

        smpl_layer_lhand = SMPLLayer(
            model_type="mano",
            use_pca=False,  # v_template=sbj_vtemp_lhand.vertices,
            flat_hand_mean=True,
            is_rhand=False,
            num_pca_comps=48,
        ).to('cpu')
        smpl_layer_rhand = SMPLLayer(
            model_type="mano",
            use_pca=False,  # v_template=sbj_vtemp_rhand.vertices,
            flat_hand_mean=True,
            is_rhand=True,
            num_pca_comps=48,
        ).to('cpu')

        mano_layer = ManoLayer(
            mano_root="data/body_models/mano",
            use_pca=False,
            flat_hand_mean=False,
        )

        body_pose = (
            R.from_matrix(
                sbj_pose["body_pose"].detach().numpy().reshape(-1, 3, 3)
            )
            .as_rotvec()
            .reshape(-1, 63)
        )
        left_hand_pose = (
            R.from_matrix(
                sbj_pose["left_hand_pose"].detach().numpy().reshape(-1, 3, 3)
            )
            .as_rotvec()
            .reshape(-1, 45)
        )
        right_hand_pose = (
            R.from_matrix(
                sbj_pose["right_hand_pose"].detach().numpy().reshape(-1, 3, 3)
            )
            .as_rotvec()
            .reshape(-1, 45)
        )

        seq = SMPLSequence(
            body_pose,
            smpl_layer,
            trans=sbj_trans,
            poses_root=sbj_orient,
            poses_left_hand=left_hand_pose,
            poses_right_hand=right_hand_pose,
            z_up=True,
        )

        T = seq.vertices.shape[0]
        T1 = 5 * T
        verts_x_interp = np.zeros((T1, seq.vertices.shape[1]))
        verts_y_interp = np.zeros((T1, seq.vertices.shape[1]))
        verts_z_interp = np.zeros((T1, seq.vertices.shape[1]))
        x = np.linspace(0, T - 1, T)
        x_new = np.linspace(0, T - 1, T1)
        for v1 in range(0, seq.vertices.shape[1]):
            verts_sbj_x = seq.vertices[:, v1, 0]
            verts_sbj_y = seq.vertices[:, v1, 1]
            verts_sbj_z = seq.vertices[:, v1, 2]
            f_x = interpolate.interp1d(x, verts_sbj_x, kind="linear")
            f_y = interpolate.interp1d(x, verts_sbj_y, kind="linear")
            f_z = interpolate.interp1d(x, verts_sbj_z, kind="linear")
            verts_x_interp[:, v1] = f_x(x_new)
            verts_y_interp[:, v1] = f_y(x_new)
            verts_z_interp[:, v1] = f_z(x_new)
        verts_x_interp = torch.from_numpy(verts_x_interp).unsqueeze(2)
        verts_y_interp = torch.from_numpy(verts_y_interp).unsqueeze(2)
        verts_z_interp = torch.from_numpy(verts_z_interp).unsqueeze(2)
        verts_interp = torch.cat(
            (verts_x_interp, verts_y_interp, verts_z_interp), dim=-1
        )
        verts_sbj_all = torch.cat(
            (torch.from_numpy(seq.vertices[0]).unsqueeze(0), verts_interp),
            dim=0,
        ).numpy()

        # vertex_colors = np.tile([0.5,0.5,0.5,0], (verts_sbj_all.shape[0],verts_sbj_all.shape[1],1))
        # vertex_colors[:,hand_idxs_left,:] = [158/255.0, 121/255.0, 119/255.0, 1.0]
        # vertex_colors[:,hand_idxs_right,:] = [121/255.0, 119/255.0, 158/255.0, 1.0]

        seq_pc = Meshes(
            verts_sbj_all, seq.faces, color=obj_mesh_color#, z_up=True,
        )

        # faces_dict = {}
        # faces_dict['mano_faces_left'] = smpl_layer_lhand.bm.faces
        # faces_dict['mano_faces_right'] = smpl_layer_rhand.bm.faces
        # import pdb; pdb.set_trace()
        # np.save('dataset/MANO_FACES.npy', faces_dict)

        seq_mano_l = Meshes(
            verts_sbj_all[:, hand_idxs_left],
            smpl_layer_lhand.bm.faces,
            # z_up=True,
            color=hand_l_mesh_color,
        )
        seq_mano_r = Meshes(
            verts_sbj_all[:, hand_idxs_right],
            smpl_layer_rhand.bm.faces,
            # z_up=True,
            color=hand_r_mesh_color,
        )

        full_pose_rotmat = (
            R.from_rotvec(
                np.concatenate(
                    [
                        body_pose,
                        np.zeros((15, 9)),
                        left_hand_pose,
                        right_hand_pose,
                    ],
                    axis=1,
                ).reshape(-1, 3)
            )
            .as_matrix()
            .reshape(15, -1, 3, 3)
        )
        # glob_rots = compute_global_orientations(full_pose_rotmat, smpl_layer.bm.parents)
        # import pdb; pdb.set_trace()
        right_hand_joint_id = [
            i
            for i, n in enumerate(smplx.joint_names.JOINT_NAMES)
            if n in ["right_wrist"]
        ][0]
        betas_l = np.load(
            "data/grab/tools/subject_meshes/"
            + gender
            + "/"
            + sbj_id
            + "_lhand_betas.npy",
            allow_pickle=True,
        )
        betas_r = np.load(
            "data/grab/tools/subject_meshes/"
            + gender
            + "/"
            + sbj_id
            + "_rhand_betas.npy",
            allow_pickle=True,
        )

        r_hand_joints = (
            seq.joints[:, right_hand_joint_id, :]
            - smpl_layer_rhand.bm(
                hand_pose=torch.zeros((1, 45)),
                betas=torch.tensor(betas_r, dtype=torch.float32).unsqueeze(0),
            )
            .joints[0, 0]
            .detach()
            .numpy()
        )
        left_hand_joint_id = [
            i
            for i, n in enumerate(smplx.joint_names.JOINT_NAMES)
            if n in ["left_wrist"]
        ][0]
        l_hand_joints = (
            seq.joints[:, left_hand_joint_id, :]
            - smpl_layer_lhand.bm(hand_pose=torch.zeros((1, 45)))
            .joints[0, 0]
            .detach()
            .numpy()
        )  # + np.tile([ 0.01749325,  0.00330192, -0.00645876], (len(seq.rbs._rb_pos),1))

        pc_joints = PointClouds(seq.joints, point_size=20, z_up=True)
        r_hand_joints_mano = (
            seq.joints[:, right_hand_joint_id, :]
            - mano_layer(torch.zeros((1, 48), dtype=torch.float32))[1].numpy()[
                :, 0
            ]
            / 1000.0
        )

        # glob_orient_l = R.from_matrix(seq.rbs._rb_ori[:,-2]).as_rotvec()
        # glob_orient_r = R.from_matrix(seq.rbs._rb_ori[:,-1]).as_rotvec()

        # seq_l = SMPLSequence(left_hand_pose, smpl_layer_lhand, trans=l_hand_joints,poses_root=glob_orient_l, z_up=True, betas=betas_l, is_rigged=False, color=[1,0,0,1])
        # seq_r = SMPLSequence(right_hand_pose, smpl_layer_rhand, trans=r_hand_joints, poses_root=glob_orient_r, z_up=True, betas=betas_r, is_rigged=False, color=[1,0,0,1])

        # mano_layer = mano_layer(torch.tensor(np.concatenate((glob_orient_r,right_hand_pose),axis=1),dtype=torch.float32),th_trans=torch.tensor(r_hand_joints_mano,dtype=torch.float32),
        #                         th_betas=torch.tensor(np.tile(betas_r,(glob_orient_r.shape[0],1)),dtype=torch.float32))[0]/1000.0

        # seq_r_pc = Meshes(seq_r.vertices-smpl_layer_rhand.bm(hand_pose=torch.zeros((1,45))).joints[0,0].detach().numpy()[np.newaxis,np.newaxis], seq_r.faces, z_up=True)
        # mano_orig = Meshes(mano_layer.numpy(), seq_r.faces, z_up=True)
        # v.scene.camera.fov = 22
        # v.scene.camera.target = [-0.089,0.361,0.073]
        # v.scene.camera.position = [-0.088,1.435,3.293]
        # v.scene.camera.up = [0,1.0,0]

        v.scene.add(seq_mano_l, seq_mano_r, mesh_frame)

        v.scene.camera.fov = 40
        v.scene.camera.target = [0.0, 0.218, -0.093]
        # v.scene.camera.position = [0.0,0.5,2.2]
        v.scene.camera.position = [0.0, 0.65, 2.152]
        v.scene.camera.up = [0, 1.0, 0]
        v.scene.floor.position = [0.0, -0.4, 0.0]
        v.auto_set_camera_target = False
        v.auto_set_floor = False
        v.scene.remove(v.scene.lights[0])
        v.scene.remove(v.scene.origin)

        verts_dict = {}
        verts_dict["verts_hand_l"] = verts_sbj_all[:, hand_idxs_left]
        verts_dict["verts_hand_r"] = verts_sbj_all[:, hand_idxs_right]
        verts_dict["verts_obj"] = obj_verts

        verts_dict["faces_hand_l"] = seq_mano_l.faces
        verts_dict["faces_hand_r"] = seq_mano_r.faces
        verts_dict["obj_faces"] = obj_mesh.faces
        
        data = obj_verts, obj_mesh.faces, verts_sbj_all[:, hand_idxs_right], seq_mano_r.faces, verts_sbj_all[:, hand_idxs_left], seq_mano_l.faces

        os.makedirs(export_dir, exist_ok=True)
        torch.save(data, os.path.join(export_dir, export_name))

        print(os.path.join("data", "imos_meshes", fn.split(".")[0] + ".npy"))
        # if not os.path.isfile(
        #     os.path.join("data", "imos_meshes", fn.split(".")[0] + ".npy")
        # ):
        #     np.save(
        #         os.path.join("data", "imos_meshes", fn.split(".")[0] + ".npy"),
        #         verts_dict,
        #     )

        # id_dict = {}
        # with open("data/file_names.txt", "r") as file:
        #     # Read each line in the file
        #     for line in file:
        #         # Strip any leading/trailing whitespace
        #         line = line.strip()

        #         # Split the line at the comma
        #         value, key = line.split(",", 1)

        #         # Add the key-value pair to the dictionary
        #         id_dict[key] = value

        # # Data to be written
        # data = [
        #     [
        #         id_dict["_".join(fn.split("_")[:-1])],
        #         sbj_id,
        #         obj_name,
        #         intent_name,
        #     ]
        # ]

        # file_dict = {}
        # file_dict["obj_name"] = obj_name
        # file_dict["intent_vec"] = mapping_intent[intent_name]
        # file_dict["joints"] = np.concatenate((seq_l.joints, seq_r.joints), axis = 1)
        # file_dict["global_orient_obj"] = obj_rot
        # file_dict["transl_obj"] = obj_pos
        # file_dict["sbj_id"] = sbj_id

        # out_path = os.path.join('dataset/predictions/imos_oursplit', seq_id)
        # np.save(out_path, file_dict)

        # Writing to the CSV file
        # with open('imos_prompts_imossplit.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data)

        # count = 0
        # # +str(folder_idx)+
        # v.save_video(
        #     video_dir="./"
        #     + id_dict["_".join(fn.split("_")[:-1])]
        #     + "_"
        #     + fn.split(".")[0]
        #     + ".mp4",
        #     frame_dir=in_dir.split("/")[0],
        #     output_fps=30,
        #     quality="high",
        # )

        # v.scene.remove(seq_mano_l, seq_mano_r, mesh_frame)
