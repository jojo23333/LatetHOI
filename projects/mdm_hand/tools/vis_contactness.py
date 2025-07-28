import torch
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

def calculate_contactness(vertices1, vertices2, scale=0.01):
    """ Calculate the contactness for each vertex in vertices1 against all vertices in vertices2. """
    dist_matrix = np.sqrt(((vertices1[:, None, :] - vertices2[None, :, :]) ** 2).sum(axis=2))
    min_distances = np.min(dist_matrix, axis=1)
    contactness_values = np.maximum(1 - min_distances / scale, 0)
    return contactness_values

def save_colored_mesh(file_path, vertices, faces, contactness, color_map='plasma'):
    """
    Saves a mesh as a PLY file with vertex colors based on the contactness.
    
    Parameters:
        file_path (str): Path to save the PLY file.
        vertices (np.ndarray): Vertex positions.
        faces (np.ndarray): Array of faces.
        contactness (np.ndarray): Contactness values for each vertex.
        color_map (str): Matplotlib colormap name.
    """
    cmap = plt.get_cmap(color_map)
    colors = cmap(contactness*2)[:, :3]  # Get RGB values
    colors = np.clip(colors, 0, 1)  # Ensure values are within [0, 1]

    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Save the mesh to a PLY file
    o3d.io.write_triangle_mesh(file_path, mesh)

def process_mesh_data(file_path, outname):
    # Load data from the file
    o_v, o_f, rh_v, rh_f, lh_v, lh_f = torch.load(file_path, map_location='cpu')

    # Initialize accumulators for contactness for each hand
    num_frames = o_v.shape[0]
    accumulated_rh_contactness = np.zeros_like(o_v[0,:,0])
    accumulated_lh_contactness = np.zeros_like(o_v[0,:,0])
    cnt = 0 

    for i in range(0, num_frames, 10):
        rh_contactness = calculate_contactness(o_v[i].numpy(), rh_v[i].numpy())
        lh_contactness = calculate_contactness(o_v[i].numpy(), lh_v[i].numpy())

        # Accumulate the contactness for averaging, separately for each hand
        accumulated_rh_contactness += rh_contactness
        accumulated_lh_contactness += lh_contactness
        cnt = cnt + 1

    # Average the accumulated contactness over all frames
    average_rh_contactness = accumulated_rh_contactness / cnt
    average_lh_contactness = accumulated_lh_contactness / cnt

    base_name = os.path.basename(file_path).split('.pth')[0]
    # Save the object mesh colored by average contactness of all frames for each hand
    save_colored_mesh(f'./.exps/{outname}_{base_name}_rh.ply', o_v[0], o_f, average_rh_contactness)
    save_colored_mesh(f'./.exps/{outname}_{base_name}_lh.ply', o_v[0], o_f, average_lh_contactness)

def main():
    parser = argparse.ArgumentParser(description='Process 3D mesh data and compute contactness.')
    parser.add_argument('-f', '--file', required=True, help='Path to the input .pth file containing the mesh data.')
    parser.add_argument('-n', '--name', required=True, help='Path to the input .pth file containing the mesh data.')
    
    
    args = parser.parse_args()

    process_mesh_data(args.file, args.name)

if __name__ == '__main__':
    main()