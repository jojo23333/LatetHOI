import torch
import numpy as np
# from torch_scatter import scatter
from torch_geometric.nn.pool import fps


def voxelize(vertex_feat, discrete_coord):
    key = fnv_hash_vec(discrete_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
    idx_unique = idx_sort[idx_select]
    discrete_coord = discrete_coord[idx_unique]
    vertex_feat = vertex_feat[idx_unique]
    return vertex_feat, discrete_coord

def get_unique_idx(vertex_feat, discrete_coord):
    key = fnv_hash_vec(discrete_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
    idx_unique = idx_sort[idx_select]
    return idx_unique

def fps_batch(vertex_feat, normal, pos, fps_ratio):
    coord = pos[0]
    x = torch.from_numpy(coord).cuda()
    batch = torch.zeros(coord.shape[0], dtype=torch.int64).cuda()
    idx = fps(x, batch, fps_ratio).cpu().numpy()
    return vertex_feat[:, idx], normal[:, idx], pos[:, idx], pos[:, idx]

def voxelize_batch(vertex_feat, normal, pos, voxel_size):
    discrete_coord = np.floor(pos / np.array(voxel_size)).astype(np.int32)
    discrete_coord -= discrete_coord.reshape(-1, 3).min(0)
    idx= get_unique_idx(vertex_feat[0], discrete_coord[0])
    return vertex_feat[:, idx], normal[:, idx], discrete_coord[:, idx], pos[:, idx]
    # feat = []
    # coord = []
    # for i in range(vertex_feat.shape[0]):
    #     feat_i, coord_i = voxelize(vertex_feat[i], discrete_coord[i])
    #     feat.append(feat_i)
    #     coord.append(coord_i)
    # import 
    # return np.stack(feat, axis=0), np.stack(coord, axis=0)

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr
