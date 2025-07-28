#/bin/bash
# conda activate hand
CUDA_VISIBLE_DEVICES=0 python grab/grab_preprocessing_adapt_flat_hand.py --filter s1/ > ../../.exps/output1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python grab/grab_preprocessing_adapt_flat_hand.py --filter s2/ > ../../.exps/output2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python grab/grab_preprocessing_adapt_flat_hand.py --filter s3/ > ../../.exps/output3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python grab/grab_preprocessing_adapt_flat_hand.py --filter s4/ > ../../.exps/output4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python grab/grab_preprocessing_adapt_flat_hand.py --filter s5/ > ../../.exps/output5.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python grab/grab_preprocessing_adapt_flat_hand.py --filter s6/ > ../../.exps/output6.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python grab/grab_preprocessing_adapt_flat_hand.py --filter s7/ > ../../.exps/output7.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python grab/grab_preprocessing_adapt_flat_hand.py --filter s8/ > ../../.exps/output8.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python grab/grab_preprocessing_adapt_flat_hand.py --filter s9/ > ../../.exps/output9.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python grab/grab_preprocessing_adapt_flat_hand.py --filter s10/ > ../../.exps/output10.log 2>&1 &