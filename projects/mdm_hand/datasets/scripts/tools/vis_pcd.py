import os
import open3d as o3d
import numpy as np
import torch

def cart2sph(xyz_tensor):
    assert xyz_tensor.shape[-1] == 3, "Input tensor must have a shape of (..., 3)"
    
    r = torch.sqrt(torch.sum(xyz_tensor ** 2, dim=-1))
    theta = torch.acos(xyz_tensor[..., 2] / r)
    phi = torch.atan2(xyz_tensor[..., 1], xyz_tensor[..., 0])
    
    return r, theta, phi

def unit_normal(normal):
    r = torch.sqrt(torch.sum(normal ** 2, dim=-1, keepdim=True))
    return normal / (r + 1e-8)

def sph2cart(normal_tensor):
    theta, phi = normal_tensor[..., 0] * torch.pi, (normal_tensor[..., 1]-1) * torch.pi
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def export_mesh(coord, normal, file_path):
    phi, theta = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    r = np.sin(theta) * np.cos(phi) * 0.5 + 0.5
    g = np.sin(theta) * np.sin(phi) * 0.5 + 0.5
    b = np.cos(theta) * 0.5 + 0.5
    unit_color = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    
    if normal.shape[-1] == 3:
        color = normal
    else:
        color = sph2cart(normal)
    color = (color + 1) / 2

    coord_max = np.array([[0.5,0.5,0.5]])
    # coord_max = coord.max(dim=0, keepdim=True)[0].cpu().numpy()
    save_point_cloud(   
        np.concatenate([coord.cpu().numpy(), unit_color*0.1+coord_max], axis=0),
        np.concatenate([color.cpu().numpy(), unit_color], axis=0),
        file_path=file_path
    )
    # coord_to = coord + normal * 0.1
    # coord_all = torch.cat([coord, coord_to], dim=0)
    # cur_edge = torch.stack([torch.arange(coord.shape[0]), torch.arange(coord.shape[0])+coord.shape[0]], dim=-1)
    # save_lines(coord_all, cur_edge, color, file_path=file_path)
    
    # if file_path.rsplit("_", 1)[1] != "input":
    #     color = (feat[:, -3:] + 1) * 127.5
    #     file_path = f"./exps/debug/vis_wm/{file_path}_mesh.ply"
    #     save_lines(coord, edges.transpose(0, 1), color, file_path=file_path)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(bboxes_corners, color=(1., 0., 0.), file_path="bbox.ply", logger=None):
    bboxes_corners = to_numpy(bboxes_corners)
    # point list
    points = bboxes_corners.reshape(-1, 3)
    # line list
    box_lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 0],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    lines = []
    for i, _ in enumerate(bboxes_corners):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)
    # color list
    color = np.array([color for _ in range(len(lines))])
    # generate line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def save_lines(points, lines, color=(1., 0., 0.), file_path="lines.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    points = to_numpy(points)
    lines = to_numpy(lines)
    if isinstance(color, (list, tuple)):
        colors = np.array([color for _ in range(len(lines))])
    else:
        colors = to_numpy(color).astype(np.uint8)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Lines to: {file_path}")

def cart2sph(xyz_tensor):
    assert xyz_tensor.shape[-1] == 3, "Input tensor must have a shape of (..., 3)"
    
    r = torch.sqrt(torch.sum(xyz_tensor ** 2, dim=-1))
    theta = torch.acos(xyz_tensor[..., 2] / r)
    phi = torch.atan2(xyz_tensor[..., 1], xyz_tensor[..., 0])
    
    return r, theta, phi

def unit_normal(normal):
    r = torch.sqrt(torch.sum(normal ** 2, dim=-1, keepdim=True))
    return normal / (r + 1e-8)

def sph2cart(normal_tensor):
    theta, phi = normal_tensor[..., 0] * torch.pi, (normal_tensor[..., 1]-1) * torch.pi
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def save_point_cloud_label(coord, label=None, file_path="pc.ply", logger=None):
    color_tensor = torch.ones((256, 3), dtype=np.float)
    for k in SCANNET_COLOR_MAP_20:
        color_tensor[k] = torch.tensor(SCANNET_COLOR_MAP_20[k]) / 255.
    color = color_tensor[label, :]
    save_point_cloud(coord, color, file_path, logger)