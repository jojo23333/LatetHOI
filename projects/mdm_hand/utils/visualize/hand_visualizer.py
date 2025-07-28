import cv2
import os
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from io import BytesIO

def normalize_2d_pose(pose_2d, s):

    # import ipdb; ipdb.set_trace()
    min_s = np.min(pose_2d)
    max_s = np.max(pose_2d)

    scale = (s - 1) / (max_s - min_s)
    pose = (pose_2d-min_s) * scale
    return pose

    # Get the min and max values of x and y coordinates
    min_x, min_y = np.min(pose_2d[..., :2], axis=(0, 1))
    max_x, max_y = np.max(pose_2d[..., :2], axis=(0, 1))

    # Compute the scaling factors for x and y
    scale_x = (w - 1) / (max_x - min_x)
    scale_y = (h - 1) / (max_y - min_y)

    # Apply the scaling and translation to normalize the coordinates
    pose_2d[..., 0] = (pose_2d[..., 0] - min_x) * scale_x
    pose_2d[..., 1] = (pose_2d[..., 1] - min_y) * scale_y
    return pose_2d

DEFATUL_EXTRINSIC = [
    [-1/sqrt(2), 1/sqrt(2), 0, 0],
    [-1/sqrt(6), -1/sqrt(6), 2/sqrt(6), 0],
    [-1/sqrt(3), -1/sqrt(3), -1/sqrt(3), -1],
]

DEFATUL_INTRINSIC = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

class HandPoseVisualizer:
    def __init__(self, intrinsic=DEFATUL_INTRINSIC, frame_size=(600, 600), fps=30):
        self.intrinsic = intrinsic
        self.frame_size = frame_size
        self.fps = fps

    def create_cameras(self, front_extrinsic):
        # Define rotations for top, left, right views
        top_rotation = cv2.Rodrigues(np.array([-np.pi / 2, 0, 0]))[0]
        left_rotation = cv2.Rodrigues(np.array([0, -np.pi / 2, 0]))[0]
        right_rotation = cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0]

        return {
            'front': {'intrinsic': self.intrinsic, 'extrinsic': front_extrinsic},
            'top': {'intrinsic': self.intrinsic, 'extrinsic': top_rotation @ front_extrinsic},
            'left': {'intrinsic': self.intrinsic, 'extrinsic': left_rotation @ front_extrinsic},
            'right': {'intrinsic': self.intrinsic, 'extrinsic': right_rotation @ front_extrinsic}
        }

    def project_3d_to_2d(self, points_3d, intrinsic, extrinsic):
        homogenous_points = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=-1)

        points_2d_homogenous = (intrinsic @ (extrinsic @ homogenous_points.T)).T
        points_2d = points_2d_homogenous[:, :2] / points_2d_homogenous[:, 2:3]
        return points_2d
    
    def project_3d(self, points_3d, extrinsic):
        homogenous_points = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=-1)
        points_3d_homogenous = (extrinsic @ homogenous_points.T).T
        return points_3d_homogenous[:, :2]


    def render_skeleton(self, points_2d):
        image = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
        color_map = cm.get_cmap('jet')

        # Keypoint hierarchy
        hierarchy = [(0, 1), (1, 2), (2, 3), (3, 20), (4, 5), (5, 6), (6, 7), (7, 20), (8, 9),
                     (9, 10), (10, 11), (11, 20), (12, 13), (13, 14), (14, 15), (15, 20), (16, 17),
                     (17, 18), (18, 19), (19, 20), (21, 22), (22, 23), (23, 24), (24, 41), (25, 26),
                     (26, 27), (27, 28), (28, 41), (29, 30), (30, 31), (31, 32), (32, 41), (33, 34),
                     (34, 35), (35, 36), (36, 41), (37, 38), (38, 39), (39, 40), (40, 41)]

        for start, end in hierarchy:
            color = tuple((255 * np.array(color_map(end / 41)))[:3].astype(int))
            color = (int(color[0]), int(color[1]), int(color[2]))
            # print(color, type(color), type(color[0]))
            # import ipdb; ipdb.set_trace()
            cv2.line(image, tuple(points_2d[start].astype(int)), tuple(points_2d[end].astype(int)), color, 2)

        return image

    def export_hand_video(self, hand_sequence, output_file, front_extrinsic=DEFATUL_EXTRINSIC):
        # Create cameras
        cameras = self.create_cameras(front_extrinsic)

        # Get unified scale and offset
        scale, offset = self.get_unified_scale_and_offset(hand_sequence, cameras)

        # Create video writer
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.frame_size[1] * 2, self.frame_size[0] * 2))

        # Iterate through the frames
        for frame_3d in tqdm(hand_sequence):
            frame_grid = np.zeros((self.frame_size[0] * 2, self.frame_size[1] * 2, 3), dtype=np.uint8)

            # Render each view and place in the grid
            for i, (view, camera) in enumerate(cameras.items()):
                points_2d = self.project_3d_to_2d(frame_3d, camera['intrinsic'], camera['extrinsic'])
                points_2d = points_2d * scale + offset  # Apply scaling and offset
                image_view = self.render_skeleton(points_2d)
                row = i // 2
                col = i % 2
                frame_grid[row * self.frame_size[0]:(row + 1) * self.frame_size[0],
                           col * self.frame_size[1]:(col + 1) * self.frame_size[1]] = image_view

            video_writer.write(frame_grid)

        # Release the video writer
        video_writer.release()

        print(f'Video exported to {output_file}')

    def export_hand_video_fixed_view(self, hand_sequence, output_file, front_extrinsic=DEFATUL_EXTRINSIC, fps=30):
        # Create cameras
        cameras = self.create_cameras(front_extrinsic)

        # Get unified scale and offset
        scale, offset = self.get_unified_scale_and_offset(hand_sequence, cameras)

        # Create video writer
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.frame_size[1] * 2, self.frame_size[0] * 2))

        # Iterate through the frames
        points_2d = [[], [], [], []]
        points_2d = [np.stack([self.project_3d(frame_3d, camera['extrinsic']) for camera in cameras.values()], axis=0)
                     for frame_3d in hand_sequence]
        points_2d = np.stack(points_2d, axis=0)
        points_2d = normalize_2d_pose(points_2d, self.frame_size[0])
        
        for fid in range(points_2d.shape[0]):
            frame_grid = np.zeros((self.frame_size[0] * 2, self.frame_size[1] * 2, 3), dtype=np.uint8)
            points_2d_cameras = points_2d[fid]
            # Render each view and place in the grid
            for i, (view, camera) in enumerate(cameras.items()):
                image_view = self.render_skeleton(points_2d_cameras[i])
                row = i // 2
                col = i % 2
                frame_grid[row * self.frame_size[0]:(row + 1) * self.frame_size[0],
                           col * self.frame_size[1]:(col + 1) * self.frame_size[1]] = image_view

            video_writer.write(frame_grid)

        # Release the video writer
        video_writer.release()

        print(f'Video exported to {output_file}')
    
    def get_matplotlib_frame(self, pose_3d, gt_pose_3d=None, mask=None, title=None, scale=(-300, 300), view_angles = [(30, -45), (30, 45), (30, 0), (30, 90)]):
        def plot_hand_3d(pose3d, gt_pose3d, mask, ax):
            color_map = cm.get_cmap('jet')

            # Keypoint hierarchy
            hierarchy = [(0, 1), (1, 2), (2, 3), (3, 20), (4, 5), (5, 6), (6, 7), (7, 20), (8, 9),
                        (9, 10), (10, 11), (11, 20), (12, 13), (13, 14), (14, 15), (15, 20), (16, 17),
                        (17, 18), (18, 19), (19, 20), (21, 22), (22, 23), (23, 24), (24, 41), (25, 26),
                        (26, 27), (27, 28), (28, 41), (29, 30), (30, 31), (31, 32), (32, 41), (33, 34),
                        (34, 35), (35, 36), (36, 41), (37, 38), (38, 39), (39, 40), (40, 41)]

            if gt_pose3d is not None:
                for start, end in hierarchy:
                    if pose_3d is not None:
                        ax.plot([gt_pose3d[start, 0], gt_pose3d[end, 0]],
                                [gt_pose3d[start, 1], gt_pose3d[end, 1]],
                                [gt_pose3d[start, 2], gt_pose3d[end, 2]], color='r', alpha=0.2)
                    elif mask is not None and (~mask[start] or ~mask[end]):
                        ax.plot([gt_pose3d[start, 0], gt_pose3d[end, 0]],
                            [gt_pose3d[start, 1], gt_pose3d[end, 1]],
                            [gt_pose3d[start, 2], gt_pose3d[end, 2]], color='r', alpha=0.2)
                    else:
                        ax.plot([gt_pose3d[start, 0], gt_pose3d[end, 0]],
                                [gt_pose3d[start, 1], gt_pose3d[end, 1]],
                                [gt_pose3d[start, 2], gt_pose3d[end, 2]], color='r')

            if pose_3d is not None:
                for start, end in hierarchy:
                    # import ipdb; ipdb.set_trace()
                    color = np.array(color_map(end/41))
                    # color = tuple((np.array(color_map(end / 41)))[:3].astype(int))
                    # color = (int(color[0]), int(color[1]), int(color[2]))
                    ax.plot([pose3d[start, 0], pose3d[end, 0]],
                            [pose3d[start, 1], pose3d[end, 1]],
                            [pose3d[start, 2], pose3d[end, 2]], color=color)  
                

        frame_grid = np.zeros((self.frame_size[0] * 2, self.frame_size[1] * 2, 3), dtype=np.uint8)
        for i ,(elev, azim) in enumerate(view_angles):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            if title is not None:
                ax.set_title(title + f' D{azim}')

            # Set view angle
            ax.view_init(elev=elev, azim=azim)

            # 设置坐标轴的范围和外观
            ax.set_xlim([scale[0][0], scale[1][0]])
            ax.set_ylim([scale[0][1], scale[1][1]])
            ax.set_zlim([scale[0][2], scale[1][2]])

            # 绘制手部姿态
            plot_hand_3d(pose_3d, gt_pose_3d, mask, ax)
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100)  # dpi=100 for 600x600 resolution
            buf.seek(0)
            
            # Convert buffer to OpenCV image
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)
            buf.close()
            plt.close(fig)
            
            # Convert from RGB to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            row = i // 2
            col = i % 2
            frame_grid[row * self.frame_size[0]:(row + 1) * self.frame_size[0],
                        col * self.frame_size[1]:(col + 1) * self.frame_size[1]] = img_bgr

        return frame_grid 

    def get_frame(self, hand_sequence, title, scale, gt_sequence=None, mask=None, view_angles=[(30, -45), (30, 45), (30, 0), (30, 90)]):
        len_video = hand_sequence.shape[0] if hand_sequence is not None else gt_sequence.shape[0]
        for fid in tqdm(range(len_video)):
            mask_frame = None if mask is None else mask[fid]
            gt_frame = None if gt_sequence is None else gt_sequence[fid]
            pose_frame = None if hand_sequence is None else hand_sequence[fid]
            yield self.get_matplotlib_frame(pose_frame, gt_pose_3d=gt_frame, mask=mask_frame, title=title, scale=scale, view_angles=view_angles)

    def export_vis(self, hand_sequence, output_file, gt_sequence=None, mask=None, title=['Generate', "GT"], gt_column=True, fps=30):
        num_sequence = len(hand_sequence) + 1
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.frame_size[1]*2*num_sequence, self.frame_size[0]*2))
        hand_sequence_all = np.concatenate(hand_sequence, axis=0)
        scale_max = np.max(hand_sequence_all, axis=(0, 1))
        scale_min = np.min(hand_sequence_all, axis=(0, 1))
        for frames in zip(
            *([self.get_frame(s, title[i], (scale_min, scale_max), gt_sequence)
              for i, s in enumerate(hand_sequence)
            ] + ([
                self.get_frame(None, 'GT', (scale_min, scale_max), gt_sequence, mask)
            ] if gt_column else []))
        ):
            frame_grid = np.zeros((self.frame_size[0] * 2, self.frame_size[1] * 2 * num_sequence, 3), dtype=np.uint8)
            for i, frame in enumerate(frames):
                frame_grid[:, self.frame_size[1]*2*i:self.frame_size[1]*2*(i+1), :] = frame
            frame_grid = frame_grid[..., [2,1,0]]
            video_writer.write(frame_grid)
        video_writer.release()
        print(f'Video exported to {output_file}')
    
    def export_simple_vis(self, hand_sequence, output_file):
         # Create video writer
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.frame_size[1]*2, self.frame_size[0]*2))
        scale_max = np.max(hand_sequence, axis=(0, 1))
        scale_min = np.min(hand_sequence, axis=(0, 1))
        for frame in self.get_frame(hand_sequence, "4 VIEWS", (scale_min, scale_max), view_angles=[(90, -90), (-90, -90), (0, -90), (0, -0)]):
            frame = frame[..., [2,1,0]]
            video_writer.write(frame)
        video_writer.release()
        print(f'Video exported to {output_file}')


if __name__ == '__main__':
    import json
    with open('/mnt/graphics_ssd/nimble/datasets/AssemblyHands/cvpr_release/annotations/val/assemblyhands_val_ego_calib_v1-1.json', 'r') as f:
        calib = json.load(f)
    with open('/mnt/graphics_ssd/nimble/datasets/AssemblyHands/cvpr_release/annotations/val/assemblyhands_val_ego_data_v1-1.json', 'r') as f:
        data = json.load(f)  
    with open('/mnt/graphics_ssd/nimble/datasets/AssemblyHands/cvpr_release/annotations/val/assemblyhands_val_joint_3d_v1-1.json', 'r') as f:
        joint = json.load(f)
    print(calib['calibration'].keys())
    for i in range(len(calib["calibration"].keys())):
        seq_name = list(calib["calibration"].keys())[i]
        cam_name = list(calib["calibration"][seq_name]["intrinsics"].keys())[i]
        intrinsic = calib["calibration"][seq_name]["intrinsics"][cam_name]
        frame_ids = list(joint["annotations"][seq_name].keys())
        # frame_ids = list(calib["calibration"][seq_name]["extrinsics"].keys())
        extrinsic = calib["calibration"][seq_name]["extrinsics"][frame_ids[0]][cam_name]

        # # Example usage
        # intrinsic = [
        #     [133.74160766601562, 0.0, 321.2432556152344],
        #     [0.0, 133.86346435546875, 237.78297424316406],
        #     [0.0, 0.0, 1.0]
        # ]
        intrinsic = np.array(intrinsic)
        extrinsic = np.array(extrinsic)
        hand_sequence = np.array([np.array(joint["annotations"][seq_name][fid]["world_coord"]) for fid in frame_ids[:2000]])
        # hand_sequence = np.stack(hand_sequence, axis=0)
        # extrinsic = [
        #     [0.6752108335494995, 0.10247515141963959, 0.7304719686508179, 137.20330810546875],
        #     [0.06562834233045578, -0.9947212934494019, 0.07888227701187134, 495.4915771484375],
        #     [0.7346994876861572, -0.005322529934346676, -0.6783718466758728, 408.6869201660156]
        # ]
        from IPython import get_ipython
        from ipdb.__main__ import _get_debugger_cls
        import ipdb
        import os
        import torch
        debugger = _get_debugger_cls()
        shell = get_ipython()
        shell.debugger_history_file = os.path.join("/mnt/nimble/nimble-dgx/users/muchenli/debug", '.pdbhistory')

        visualizer = HandPoseVisualizer()
        for fps in [5, 10]:
            for j in range(5):
                start = j * 60 * 30 // fps
                end = (j + 1) * 60 * 30 // fps
                seq1 = hand_sequence[start:end:30//fps]
                seq2 = hand_sequence[start+5:end+5:30//fps]
                bp = torch.ones(*hand_sequence.shape[:-1]) * 0.5
                mask = torch.torch.bernoulli(bp).to(torch.bool).numpy()
                visualizer.export_vis([seq1],f'./exps/debug/{i}_{j}_hand_pose_{fps}.mp4', gt_sequence=seq2, mask=mask, title=['generate', 'GT'], fps=fps)
                # visualizer.export_hand_video_fixed_view(hand_sequence[start:end:30//fps], f'./exps/debug/{i}_{j}_hand_pose_{fps}.mp4', fps=fps)
        # visualizer.export_simple_vis(hand_sequence, './exps/debug/hand_pose_1.mp4')
