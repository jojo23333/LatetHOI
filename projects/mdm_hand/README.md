# LatentHOI: On the Generalizable Hand Object Motion Generation with Latent Hand Diffusion

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://jojoml.github.io/latentHOI/)
[![Paper](https://img.shields.io/badge/CVPR-2025-red)](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_LatentHOI_On_the_Generalizable_Hand_Object_Motion_Generation_with_Latent_CVPR_2025_paper.pdf)
[![Poster](https://img.shields.io/badge/Poster-PDF-green)](https://drive.google.com/file/d/1qKdHovBW6kdGMl_u5pVHifSGCANAfx7P/view?usp=sharing)


## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)

## TODO
- [ ] Code cleanup and refactoring
- [ ] Upload pretrained checkpoints to HuggingFace
- [ ] Integrate motionrender for visualization

## Environment Setup
For environment setup instructions, please refer to [projects/mdm_hand/environment.md](projects/mdm_hand/environment.md).

## Data Preparation

### Hand Model (MANO)
#### Step 1: Register and Download
1. Register for a [MANO account](https://mano.is.tue.mpg.de/).
2. Download the checkpoint files.
#### Step 2: File Placement
Place the downloaded files under `mdm_hand/data/body_models/mano`.
#### Expected Folder Structure
The directory structure should look like this:
```
data/
└── body_models/
    └── mano/
        ├── MANO_LEFT.pkl
        └── MANO_RIGHT.pkl
```

### Download processed dataset
For easy use of the code the processed data, it can be downloaded from [huggingface](https://huggingface.co/datasets/jojo23333/LatentHOI-data)
Run following script:
```
cd projects/mdm_hand/data
wget https://huggingface.co/datasets/jojo23333/LatentHOI-data/resolve/main/grab_frames.tar.gz
wget https://huggingface.co/datasets/jojo23333/LatentHOI-data/resolve/main/grab_seq20fps.tar.gz
tar -xzvf grab_frames.tar.gz
tar -xzvf grab_seq20fps.tar.gz
```

### (Alternatively) Prepare from Raw
All data preparation scripts should be run from the `projects/mdm_hand/datasets/GRAB` directory.

**GRAB Dataset**
```python
# For VAE training (single frame hand data, left hand not in contact are omitted)
python grab/grab_preprocessing_adapt_flat_hand.py

# For Diffusion model training (sequence data)
python grab/grab_preprocessing_all_seq.py
```

**DexYCB Dataset**
```python
# For VAE training
python grab/dexycb_preprocessing_all_seq.py

# For Diffusion model training (with --seq flag)
python grab/dexycb_preprocessing_all_seq.py --seq
```
<!-- ### GraspXL Dataset
```python
python grab/graspxl_preprocessing.py
``` -->

## Training
### GraspVAE 
```python
# GraspXL
python -m tools.train_vae --num-gpus 3 --resume --config config/grab/VAE_graspXL.yaml OUTPUT_DIR .exps/graspXL_vae TRAINING.BATCH_SIZE 320

# GRAB
python -m tools.train_vae --num-gpus 1 --resume --config config/VAE/VAE_grab.yaml

# DexYCB
python -m tools.train_vae --num-gpus 1 --resume --config config/grab/dexycb.yaml
```

### Latent Diffusion
```python
# GRAB
python -m tools.train_diff --num-gpus 2 --mode ldm --resume --config config/grab/LDM_pretrain_vae_AUG+.yaml  TRAINING.BATCH_SIZE 48

# DexYCB
python -m tools.train_diff -m ldm --num-gpus 1 --resume --config config/dexycb/DEX_LDM.yaml  OUTPUT_DIR .exps/LDM_DEXYCB_AUG+  TRAINING.BATCH_SIZE 32 DATASET.NUM_WORKERS 4
```

## Evaluation
```python
# Basic evaluation
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750/

# With visualization (generates videos)
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750/ --vis

# With physics evaluation
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750/ --eval

# For DexYCB dataset
eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750/ --dex
```

## Visualization
Visualization is integrated into the evaluation process. Use the `--vis` flag with the evaluation command to generate videos of the hand motions:
```python
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750/ --vis
```

## Troubleshooting
### OpenGL Headless Rendering Issue
Modify the AITViewer backend to use EGL in `aitviewer/viewer.py` line 129:
```python
self.window = base_window_cls(
    title=title,
    size=size,
    fullscreen=C.fullscreen,
    resizable=C.resizable,
    gl_version=self.gl_version,
    aspect_ratio=None,
    vsync=C.vsync,
    samples=self.samples,
    cursor=True,
    backend="egl"
)
```

### FFMPEG Video Export Issue
The provided ffmpeg might not recognize presets in commands. Solution options:
1. Download and replace the conda environment ffmpeg as described in [StyleSDF issue #20](https://github.com/royorel/StyleSDF/issues/20)
2. Use the system's global ffmpeg installation

### Memory Leak in Headless Rendering
As reported in [AITViewer issue #53](https://github.com/eth-ait/aitviewer/issues/53), use sub-processes to run rendering commands.

### ValueError: bad value(s) in fds_to_keep
This error occurs when storing shared tensors for dataloader workers. Solution:
```python
mean_latent, std_latent = copy.deepcopy(torch.chunk(mean_latent, 2, dim=-1))
dataset.mean_latent, dataset.std_latent = mean_latent.numpy(), std_latent.numpy()
```
Adding `.numpy()` converts tensors to numpy arrays, solving the shared tensor issue.


## Citation
```bibtex
@InProceedings{Muchen_LatentHOI,
    author    = {Li, Muchen and Christen, Sammy and Wan, Chengde and Cai, Yujun and Liao, Renjie and Sigal, Leonid and Ma, Shugao},
    title     = {LatentHOI: On the Generalizable Hand Object Motion Generation with Latent Hand Diffusion.},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17416-17425}
}
```