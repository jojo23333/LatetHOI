#/bin/bash
# conda activate hand
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s1/ > ../../.exps/output1.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s2/ > ../../.exps/output2.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s3/ > ../../.exps/output3.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s4/ > ../../.exps/output4.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s5/ > ../../.exps/output5.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s6/ > ../../.exps/output6.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s7/ > ../../.exps/output7.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s8/ > ../../.exps/output8.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s9/ > ../../.exps/output9.log 2>&1 &
# srun -p edith --gres=gpu:1 --time=0-03:00 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s10/ > ../../.exps/output10.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s1/ > ../../.exps/output1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s2/ > ../../.exps/output2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s3/ > ../../.exps/output3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s4/ > ../../.exps/output4.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s5/ > ../../.exps/output5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s6/ > ../../.exps/output6.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s7/ > ../../.exps/output7.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s8/ > ../../.exps/output8.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s9/ > ../../.exps/output9.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 taskset -c 12 python grab/grab_preprocessing_all_seq.py --filter s10/ > ../../.exps/output10.log 2>&1 &