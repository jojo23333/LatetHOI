
# TODO VIS for DEXYCB
# python -m tools.train_diff --config-file .exps/SELECTED_RESULTS/DEXYCB/DEX_MDM_05_13/config.yaml --mode mdm --resume --eval-only MODEL.WEIGHTS .exps/SELECTED_RESULTS/DEXYCB/DEX_MDM_05_13/model_final.pth OUTPUT_DIR .exps/SELECTED_RESULTS/DEXYCB/DEX_MDM_05_13/ TEST.BATCH_SIZE 4
# rm .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+/vis/60000
# python -m tools.eval_motion -f .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+/vis/60000 --vis --eval
# rm .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+/vis/49999
# python -m tools.eval_motion -f .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+/vis/49999 --vis --eval
# rm .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+/vis/39999
# python -m tools.eval_motion -f .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+/vis/39999 --vis --eval
# python -m tools.train_diff --config-file .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_pretrained_AUG+/config.yaml --mode ldm --resume --eval-only MODEL.WEIGHTS .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_pretrained_AUG+/model_final.pth OUTPUT_DIR .exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_pretrained_AUG+ TEST.BATCH_SIZE 4

export WDIR='.exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_pretrained_AUG+'
# CUDA_VISIBLE_DEVICES=0 python -m tools.train_diff --config-file ${WDIR}/config.yaml --resume --mode ldm --eval-only MODEL.WEIGHTS $WDIR/model_0059999.pth OUTPUT_DIR $WDIR
# mv $WDIR/vis/0 $WDIR/vis/59999
# python -m tools.eval_motion -f $WDIR/vis/59999 --vis --eval --dex

CUDA_VISIBLE_DEVICES=0 python -m tools.train_diff --config-file ${WDIR}/config.yaml --resume --mode ldm --eval-only MODEL.WEIGHTS $WDIR/model_0049999.pth OUTPUT_DIR $WDIR
mv $WDIR/vis/0 $WDIR/vis/49999
python -m tools.eval_motion -f $WDIR/vis/49999 --vis --eval --dex

CUDA_VISIBLE_DEVICES=0 python -m tools.train_diff --config-file ${WDIR}/config.yaml --resume --mode ldm --eval-only MODEL.WEIGHTS $WDIR/model_0039999.pth OUTPUT_DIR $WDIR
mv $WDIR/vis/0 $WDIR/vis/39999
python -m tools.eval_motion -f $WDIR/vis/39999 --vis --eval --dex

export WDIR='.exps/SELECTED_RESULTS/DEXYCB/DEX_LDM_AUG+'
CUDA_VISIBLE_DEVICES=0 python -m tools.train_diff --config-file ${WDIR}/config.yaml --resume --mode ldm --eval-only MODEL.WEIGHTS $WDIR/model_0059999.pth OUTPUT_DIR $WDIR
mv $WDIR/vis/0 $WDIR/vis/59999
python -m tools.eval_motion -f $WDIR/vis/59999 --vis --eval --dex

CUDA_VISIBLE_DEVICES=0 python -m tools.train_diff --config-file ${WDIR}/config.yaml --resume --mode ldm --eval-only MODEL.WEIGHTS $WDIR/model_0054999.pth OUTPUT_DIR $WDIR
mv $WDIR/vis/0 $WDIR/vis/54999
python -m tools.eval_motion -f $WDIR/vis/54999 --vis --eval --dex

CUDA_VISIBLE_DEVICES=0 python -m tools.train_diff --config-file ${WDIR}/config.yaml --resume --mode ldm --eval-only MODEL.WEIGHTS $WDIR/model_0049999.pth OUTPUT_DIR $WDIR
mv $WDIR/vis/0 $WDIR/vis/49999
python -m tools.eval_motion -f $WDIR/vis/49999 --vis --eval --dex