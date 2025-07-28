import argparse
import os
import numpy as np
import torch
import glob
import smplx.joint_names
import subprocess
import concurrent.futures
from torch.nn.functional import pdist
from tqdm import tqdm
import json
from engine.evaluation.eval_mdm import average_dicts
from math import sqrt, isnan

PYTHON_PATH = subprocess.check_output("which python", shell=True).decode('utf-8').strip()
print("CURRENT PYTHON PATH: ", PYTHON_PATH)

def get_files(path):
    pth_files = glob.glob(os.path.join(path, '*.pth'))
    file_cnt = len(pth_files)
    txt2motion = {}
    for f in pth_files:
        action_name = '_'.join(os.path.basename(f).split('_')[1:-1])
        if action_name in txt2motion:
            txt2motion[action_name].append(f)
        else:
            txt2motion[action_name] = [f]
            
    for k, v in txt2motion.items():
        print("Action Name: ", k, "Len of v", len(v))
    return pth_files, txt2motion

def run_command(cmd):
    try:
        subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error in command {' '.join(cmd)}: {e.stderr.strip()}"

def render_all(seqs, dex=False):
    if dex:
        all_commands = [
            [PYTHON_PATH, "-m", "engine.evaluation.render", "--dex", "--smooth", "-f", seq] for seq in seqs
        ]
    else:
        # all_commands = [
        #     ["/data/conda_envs/hand/bin/python", "-m", "engine.evaluation.render", "--smooth", "--obj", "-f", seq] for seq in seqs
        # ]
        all_commands = [
            [PYTHON_PATH, "-m", "engine.evaluation.render", "--smooth", "-f", seq] for seq in seqs
        ]
    max_workers = 20
    print(" ".join(all_commands[0]))
    subprocess.run(all_commands[0], text=True, capture_output=True, check=True)
    # Using ThreadPoolExecutor to manage the pool of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the run_command function to the commands
        results = list(tqdm(executor.map(run_command, all_commands), total=len(all_commands)))


def eval_phys_all(seqs, n_phys):
    seqs_to_be_run = []
    for seq in seqs:
        export_result_path = seq.replace('.pth', '.json')
        export_result_path = os.path.join(
            os.path.dirname(export_result_path), 'eval', os.path.basename(export_result_path)
        )
        if not os.path.exists(export_result_path):
            seqs_to_be_run.append(seq)
    print("Len of seqs to be run: ", len(seqs_to_be_run))
    all_commands = [
        [PYTHON_PATH, "-m", "engine.evaluation.physic_metrics", "-n", str(n_phys), "-f", seq] for seq in seqs_to_be_run
    ]
    max_workers = 4
    # Using ThreadPoolExecutor to manage the pool of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the run_command function to the commands
        results = list(tqdm(executor.map(run_command, all_commands), total=len(all_commands)))

    results = []
    for seq in seqs:
        export_result_path = seq.replace('.pth', '.json')
        export_result_path = os.path.join(
            os.path.dirname(export_result_path), 'eval', os.path.basename(export_result_path)
        )
        if not os.path.exists(export_result_path):
            print('Eval save Fail for seq: ', seq)
        else:
            with open(export_result_path, 'r') as f:
                cur_result = json.load(f)
                if isnan(cur_result['off_ground_contact_ratio']) or isnan(cur_result['off_ground_contact_ratio']):
                    print('NAN in metric of seq: ', seq)
                # elif cur_result['rhand']['inter_volume_contact'] > 1.5e-5 :
                #     continue
                else:
                    results.append(cur_result)
    result = average_dicts(results)
    print(result)
    return result
    

def eval_diversity(txt2motion, seqs):
    diversity_per_txt = {}
    all_feat = []
    for txt, seqs in txt2motion.items():
        feat = []
        for f in seqs:
            data = torch.load(f, map_location='cpu')
            if isinstance(data, dict):
                data = data['pred']
            if isinstance(data[0], np.ndarray):
                data = [torch.from_numpy(x) if x is not None else x for x in data]
            obj_v, obj_f, rhand_v, rhand_f, lhand_v, lhand_f = data
            print(f, rhand_v.shape)
            
            if lhand_v is not None:
                rhand_v = rhand_v[::10]
                T = rhand_v.shape[0]
                lhand_v = lhand_v[::10]
                feat.append(torch.cat([lhand_v.reshape(T, -1), rhand_v.reshape(T, -1)], dim=-1))
            else:
                rhand_v = rhand_v[::4]
                T = rhand_v.shape[0]
                feat.append(rhand_v.reshape(T, -1))
        
        min_t = min([x.shape[0] for x in feat])
        feat = [x[:min_t, ...] for x in feat]
        feat = torch.stack(feat, dim=0)
        # assert feat.shape[0] == repeat
        repeat = feat.shape[0]
        feat = feat.reshape(repeat, -1)
        all_feat.append(feat)
        
        if lhand_v is not None:
            diversity_per_txt[txt] = torch.mean(pdist(feat, p=2)) / sqrt(T * 778 * 2)
        else:
            diversity_per_txt[txt] = torch.mean(pdist(feat, p=2)) / sqrt(T * 778)
            
    try:
        all_feat = torch.cat(all_feat, dim=0)
        print("Total seq cnt:", all_feat.shape[0])
        
        if lhand_v is not None:
            over_all_div = torch.mean(pdist(all_feat, p=2)).item() / sqrt(T * 778 * 2)
        else:
            over_all_div = torch.mean(pdist(all_feat, p=2)).item() / sqrt(T * 778)
    except Exception as e:
        over_all_div = -1

    return diversity_per_txt, over_all_div


def main(args, dex=False):
    files, txt2motion = get_files(args.folder)
    if args.vis:
        render_all(files, dex=dex)
    if args.eval:
        print("Start Evaluating Diversity")
        diversity_per_txt, over_all_div = eval_diversity(txt2motion, files)
        print("Diversity Evaluated.")
        sd_list = list(diversity_per_txt.values())
        sample_diversity = sum(sd_list) / len(sd_list)
        print(sample_diversity.item(), over_all_div)
        result = eval_phys_all(files, args.n_phys)
        result['sample_diversity'] = sample_diversity.item()
        result['overall_diversity'] = over_all_div#.item()
        with open(os.path.join(args.folder, '.result.json'), 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script with command line arguments")
    parser.add_argument('-f', '--folder', type=str, required=True, help="The file path argument")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dex', action='store_true')
    parser.add_argument('-n', '--n_phys', type=int, default=50)
    args = parser.parse_args()
    if 'DEXYCB' in args.folder:
        print("DEXYCB use 30 frames for eval")
        args.n_phys = 30
        args.dex = True
    elif 'IMOS' in args.folder:
        print("IMOS use 15 frames for eval")
        args.n_phys = 15
    
    main(args, dex=args.dex)
