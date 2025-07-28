# Environment Setup

## Create and activate environment
```bash
conda create -n hand python=3.9 -y
conda activate hand
```

## Core dependencies
```bash
# PyTorch and related libraries
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pytorch3d -c pytorch3d --solver=classic -y
conda install pyg -c pyg -y
conda install ignite -c pytorch -y
conda install scikit-learn ffmpeg -y

# Python packages
pip install aitviewer==1.12
pip install open3d
pip install spconv-cudaxx
pip install cgal rtree
pip install ftfy regex tqdm trimesh einops pyrender ipdb easydict chumpy
pip install numpy==1.23.1 scipy
```

## Special installations
Install for CLIP (for encoding text), chamfer distance, and BPS encoding (for encoding 3d object point cloud). 
```bash
# Git repositories
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch

# SMPLX
pip install smplx[all]
```

## Visualization environment (optional)
Install visualization vitviewer based environment on macos
```bash
conda create -n "aitviewer" python=3.9 -y
conda activate aitviewer
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
python -m pip install PyQt6
AITVIEWER_INSTALL_PYQT6=1 pip install --no-binary=aitviewer aitviewer
python -m pip install chumpy numpy==1.23.1 easydict
```