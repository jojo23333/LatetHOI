import argparse
import os.path as op
import numpy as np
import random
import sys
import torch
import os
from aitviewer.configuration import CONFIG as C
from easydict import EasyDict
from engine.evaluation.render import MotionRender
try:
    from PyQt6.QtWidgets import QApplication, QFileDialog
except ImportError:
    from PyQt5.QtWidgets import QApplication, QFileDialog

C.window_type = "pyqt6"
# C.z_up = True

base_path = op.dirname(op.dirname(__file__))
print(base_path)
sys.path = ["."] + sys.path
sys.path.append(base_path)
last_dirname = '/Users/jojo/Desktop/ldm_neurips_exp/Oakink_qualitative/05_20_ldm_oakink/vis/0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_flathand", action="store_false")
    parser.add_argument("--grab", action="store_true")
    parser.add_argument("--obj", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--dex", action="store_true")
    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args

def open_file(args):
    print("PyQt6 not found, imported PyQt5 instead.")
    dialog = QFileDialog()
    global last_dirname
    file_name, _ = dialog.getOpenFileName(caption='Select file', directory=last_dirname)
    if file_name.endswith('npy'):
        data = np.load(file_name, allow_pickle=True).item()
    else:
        data = torch.load(file_name, map_location='cpu')
    print("Opening: ", file_name)
    last_dirname = os.path.dirname(file_name)
    if args.obj:
        obj_name = '_'.join(os.path.basename(file_name).split('_')[1:3])
        obj_mesh = os.path.join('/Users/jojo/Desktop/ldm_neurips_exp/OAKINK/selected_texture_uv/', obj_name, obj_name+'.obj')
    else:
        obj_mesh = None
    if isinstance(data, tuple):
        return {"pred": data}, obj_mesh
    return data, obj_mesh

def main():
    app = QApplication([])
    args = parse_args()
    random.seed(1)
    while True:
        data, obj_mesh = open_file(args)
        viewer = MotionRender(interactive=not args.headless, size=(2024, 2024), smoothing=args.smooth, dexycb=args.dex)
        viewer.render_seq(seqs=data, save_pth=False, obj_mesh=obj_mesh)
        del viewer

if __name__ == "__main__":
    main()