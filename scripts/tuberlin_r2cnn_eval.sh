cd "/content/drive/My Drive/Sketch-R2CNN/"
#CL_MODEL="vgg16"
#CL_MODEL="densenet161"
#CL_MODEL="resnext101"
#CL_MODEL="resnet101"
#CL_MODEL="resnet50attn"
CL_MODEL="resnet50"
#CL_MODEL="sketchanet"

CL_INTENSITY_CHANNELS=8

CL_RUNNAME="20-02-15_15-35-tuberlin-rnn-8c-resnet50-1epoch"
CL_CKPT_PREFIX="tuberlin_sketchanet_fold{}_iter_epoch_{}"

CL_DATASET="tuberlin"
CL_DATASET_ROOT="data/TUBerlin/TUBerlin.pkl"
CL_LOG_DIR="logs/"

CL_NOTE="Evaluation; TUBerlin dataset; Pretrained Conv on Quickdraw."

mkdir "${CL_LOG_DIR}${CL_RUNNAME}_eval"

/opt/conda/bin/python scripts/tuberlin_r2cnn_eval.py \
        --checkpoint "${CL_LOG_DIR}${CL_RUNNAME}/${CL_CKPT_PREFIX}" \
        --dataset_fn ${CL_DATASET} \
        --dataset_root "${CL_DATASET_ROOT}" \
        --intensity_channels ${CL_INTENSITY_CHANNELS} \
        --log_dir "${CL_LOG_DIR}${CL_RUNNAME}_eval" \
        --model_fn "${CL_MODEL}" \
        --note "${CL_NOTE}" \
    2>&1 | tee -a "${CL_LOG_DIR}${CL_RUNNAME}_eval/eval.log"
