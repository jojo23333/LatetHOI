# rm .exps/SELECTED_RESULTS/GRAB/GRAB_MDM/vis/93750/*.mp4
# python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_MDM/vis/93750 --vis --eval

rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/93750 --vis --eval
rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/79999/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/79999 --vis --eval
rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/59999/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_AUG+_05_15/vis/59999 --vis --eval

rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/93750/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/93750 --vis --eval
rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/89999/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/89999 --vis --eval
rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/79999/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/79999 --vis --eval
rm .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/59999/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/GRAB/GRAB_LDM_pretrained_AUG+_05_17/vis/59999 --vis --eval


rm .exps/SELECTED_RESULTS/OAKINK/oakink_ldm_05_17/vis/0/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/OAKINK/oakink_ldm_05_17/vis/0 --vis --eval
rm .exps/SELECTED_RESULTS/OAKINK/oakink_ldm_pretrained_05_17/vis/0/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/OAKINK/oakink_ldm_pretrained_05_17/vis/0 --vis --eval
rm .exps/SELECTED_RESULTS/OAKINK/oakink_mdm_05_17/vis/0/*.mp4
python -m tools.eval_motion -f .exps/SELECTED_RESULTS/OAKINK/oakink_mdm_05_17/vis/0 --vis --eval

