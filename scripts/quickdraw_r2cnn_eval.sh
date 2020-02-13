#CL_MODEL="vgg16"
#CL_MODEL="densenet161"
#CL_MODEL="resnext101"
#CL_MODEL="resnet101"
#CL_MODEL="resnet50attn"
CL_MODEL="resnet50"
#CL_MODEL="sketchanet"

CL_INTENSITY_CHANNELS=8

CL_RUNNAME="10-02_14-17-quickdraw-rnn-resnet50-1epoch"
CL_CKPT_PREFIX="quickdraw_resnet50_iter_epoch_1"

CL_DATASET="quickdraw"
CL_DATASET_ROOT="/home/craiglee/sketchrnn_processed_v1/"
CL_LOG_DIR="/media/hdd/craiglee/Data/DeepLearningSketchModeling/NeuralLineCVPR19/"

CL_NOTE="Evaluation; Quickdraw dataset;"

mkdir "${CL_LOG_DIR}${CL_RUNNAME}_eval"

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
    python quickdraw_r2cnn_eval.py \
        --checkpoint "/host${CL_LOG_DIR}${CL_RUNNAME}/${CL_CKPT_PREFIX}" \
        --dataset_fn ${CL_DATASET} \
        --dataset_root "/host${CL_DATASET_ROOT}" \
        --intensity_channels ${CL_INTENSITY_CHANNELS} \
        --log_dir "/host${CL_LOG_DIR}${CL_RUNNAME}_eval" \
        --model_fn "${CL_MODEL}" \
        --note "${CL_NOTE}" \
    2>&1 | tee -a "${CL_LOG_DIR}${CL_RUNNAME}_eval/eval.log"
