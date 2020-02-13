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

CL_INTENSITY_CHANNELS=1
CL_FUSION="late"
#CL_FUSION="early"

CL_EPOCHS=1

CL_DATASET="quickdraw"
CL_DATASET_ROOT="/home/craiglee/sketchrnn_processed_v1/"
CL_LOG_DIR="/media/hdd/craiglee/Data/DeepLearningSketchModeling/NeuralLineCVPR19/"

CL_NOTE="RNN (LSTM with dropout) + CNN + Late fusion; Pretrained CNN; Without entropy-based clean-up"

CL_RUNNAME="${CL_DATE}-${CL_DATASET}-${CL_FUSION}fusion-${CL_INTENSITY_CHANNELS}c-${CL_MODEL}-${CL_EPOCHS}epoch"
mkdir ${CL_LOG_DIR}${CL_RUNNAME}

sudo nvidia-docker run --rm \
    --network=host \
    --shm-size 8G \
    -v /:/host \
    -v /tmp/torch_extensions:/tmp/torch_extensions \
    -v /tmp/torch_models:/root/.torch \
    -w "/host$PWD" \
    -e PYTHONUNBUFFERED=x \
    -e CUDA_CACHE_PATH=/host/tmp/cuda-cache \
    py35pytorch101 \
    python quickdraw_fusion_train.py \
        --dataset_fn ${CL_DATASET} \
        --dataset_root "/host${CL_DATASET_ROOT}" \
        --fusion ${CL_FUSION} \
        --intensity_channels ${CL_INTENSITY_CHANNELS} \
        --log_dir "/host${CL_LOG_DIR}${CL_RUNNAME}" \
        --model_fn "${CL_MODEL}" \
        --note "${CL_NOTE}" \
        --num_epochs ${CL_EPOCHS} \
    2>&1 | tee -a "${CL_LOG_DIR}${CL_RUNNAME}/train.log"
