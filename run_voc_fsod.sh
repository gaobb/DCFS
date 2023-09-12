#!/usr/bin/env bash
NET=$1
NUNMGPU=$2
EXPNAME=$3
SPLIT_ID=$4

SAVEDIR=workspace/DCFS/voc-det/${EXPNAME}   #<-- change it to you path
PRTRAINEDMODEL=pretrained_models/                                     #<-- change it to you path

if [ "$NET"x = "r101"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-101.pkl
  IMAGENET_PRETRAIN_TORCH=${PRTRAINEDMODEL}/resnet101-5d3b4d8f.pth
fi

if [ "$NET"x = "r50"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-50.pkl
  IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN}/resnet50-19c8e357.pth
fi


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus ${NUNMGPU} --config-file configs/voc/dcfs_det_${NET}_base${SPLIT_ID}.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
           OUTPUT_DIR ${SAVE_DIR}/dcfs_det_${NET}_base${SPLIT_ID}


# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset voc --method remove                                    \
    --src-path ${SAVEDIR}/dcfs_det_${NET}_base${SPLIT_ID}/model_final.pth                      \
    --save-dir ${SAVEDIR}/dcfs_det_${NET}_base${SPLIT_ID}
BASE_WEIGHT=${SAVEDIR}/dcfs_det_${NET}_base${SPLIT_ID}/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #  
# --> 1. TFA-like, i.e. run seed0~9 10 times for FSOD on VOC 
classloss="DC" # "CE"
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10  
    do
        TRAIN_NOVEL_NAME=voc_2007_trainval_novel${SPLIT_ID}_${shot}shot_seed${seed}
        TEST_NOVEL_NAME=voc_2007_test_novel${SPLIT_ID}
        CONFIG_PATH=configs/voc/dcfs_fsod_${NET}_novelx_${shot}shot_seedx.yaml

        OUTPUT_DIR=${SAVEDIR}/dcfs_fsod_${NET}_novel${SPLIT_ID}/tfa-like-${classloss}/${shot}shot_seed${seed}
        BASE_WEIGHT=$OUTPUT_DIR/model_final.pth
        python3 main.py --num-gpus ${NUNMGPU} --config-file ${CONFIG_PATH}                             \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                      \
                   MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE ${classloss} \
                   DATASETS.TRAIN "('"${TRAIN_NOVEL_NAME}"',)" \
                   DATASETS.TEST  "('"${TEST_NOVEL_NAME}"',)"  \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                   TEST.PCB_MODELTYPE $NET
        #rm ${OUTPUT_DIR}/model_final.pth
    done
done
# surmarize all results
# python3 tools/extract_results.py --res-dir ${SAVE_DIR}/dcfs_fsod_${NET}_novel${SPLIT_ID}/tfa-like-${classloss} --shot-list 1 2 3 5 10  


# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVEDIR}/dcfs_det_${NET}_base${SPLIT_ID}/model_final.pth                    \
    --save-dir ${SAVEDIR}/dcfs_det_${NET}_base${SPLIT_ID}
BASE_WEIGHT=${SAVEDIR}/dcfs_det_${NET}_base${SPLIT_ID}/model_reset_surgery.pth

# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 10 times for gFSOD on voc
classloss="DC" # "CE"
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 
    do
        TRAIN_ALL_NAME=voc_2007_trainval_all${SPLIT_ID}_${shot}shot_seed${seed}  
        TEST_ALL_NAME=voc_2007_test_all${SPLIT_ID}
        CONFIG_PATH=configs/voc/dcfs_gfsod_${NET}_novelx_${shot}shot_seedx.yaml

        OUTPUT_DIR=${SAVEDIR}/dcfs_gfsod_${NET}_novel${SPLIT_ID}/tfa-like-${classloss}/${shot}shot_seed${seed} 
        python3 main.py --num-gpus ${NUNMGPU}  --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE ${classloss} \
                   DATASETS.TRAIN "('"${TRAIN_ALL_NAME}"',)" \
                   DATASETS.TEST  "('"${TEST_ALL_NAME}"',)"  \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                   TEST.PCB_MODELTYPE $NET
        #rm ${OUTPUT_DIR}/model_final.pth
    done
done
# surmarize all results
# python3 tools/extract_results.py --res-dir ${SAVE_DIR}/dcfs_gfsod_${NET}_novel${SPLIT_ID}/tfa-like-${classloss} --shot-list 1 2 3 5 10  
