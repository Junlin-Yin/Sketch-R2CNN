CL_DATE=`date '+%y-%m-%d_%H-%M'`

#CL_MODEL="vgg16"
#CL_MODEL="densenet161"
#CL_MODEL="resnext101"
#CL_MODEL="resnet101"
#CL_MODEL="resnet50attn"
CL_MODEL="resnet50"
#CL_MODEL="sketchanet"

CL_CKPT_RUNNAME=""
CL_CKPT_PREFIX=""

CL_INTENSITY_CHANNELS=8

CL_EPOCHS=200

CL_DATASET="tuberlin"
CL_DATASET_ROOT="/home/xinzhu/Projects/Sketch-R2CNN/data/TUBerlin/TUBerlin.pkl"
CL_LOG_DIR="/home/xinzhu/Projects/Sketch-R2CNN/logs/"

CL_NOTE="Pretrained model; TUBerlin dataset; RNN + CNN"

CL_RUNNAME="${CL_DATE}-${CL_DATASET}-rnn-${CL_INTENSITY_CHANNELS}c-${CL_MODEL}-${CL_EPOCHS}epoch"
mkdir ${CL_LOG_DIR}${CL_RUNNAME}

sudo docker run --rm \
    --network=host \
    --shm-size 8G \
    --gpus all \
    -v /:/host \
    -v /tmp/torch_extensions:/tmp/torch_extensions \
    -v /tmp/torch_models:/root/.torch \
    -w "/host$PWD" \
    -e PYTHONUNBUFFERED=x \
    -e CUDA_CACHE_PATH=/host/tmp/cuda-cache \
    xinzhu/sketch-grcnn \
    python tuberlin_r2cnn_train.py \
        --ckpt_nets resnet50 \
        --ckpt_prefix "/host${CL_LOG_DIR}${CL_CKPT_RUNNAME}/${CL_CKPT_PREFIX}" \
        --dataset_fn ${CL_DATASET} \
        --dataset_root "/host${CL_DATASET_ROOT}" \
        --intensity_channels ${CL_INTENSITY_CHANNELS} \
        --log_dir "/host${CL_LOG_DIR}${CL_RUNNAME}" \
        --model_fn "${CL_MODEL}" \
        --note "${CL_NOTE}" \
        --num_epochs ${CL_EPOCHS} \
    2>&1 | tee -a "${CL_LOG_DIR}${CL_RUNNAME}/train.log"
