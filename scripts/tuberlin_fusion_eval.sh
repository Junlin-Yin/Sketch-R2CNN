#CL_MODEL="vgg16"
#CL_MODEL="densenet161"
#CL_MODEL="resnext101"
#CL_MODEL="resnet101"
#CL_MODEL="resnet50attn"
CL_MODEL="resnet50"
#CL_MODEL="sketchanet"

CL_INTENSITY_CHANNELS=1
CL_FUSION="late"
#CL_FUSION="early"

CL_RUNNAME="10-12_22-51-tuberlin-rnncnnfusion-resnet50-500epoch"
CL_CKPT_PREFIX="tuberlin_resnet50_fold{}_iter_epoch_{}"

CL_DATASET="tuberlin"
CL_DATASET_ROOT="/media/hdd/craiglee/Data/TUBerlin/TUBerlin.pkl"
CL_LOG_DIR="/media/hdd/craiglee/Data/DeepLearningSketchModeling/NeuralLineCVPR19/"

CL_NOTE="Evaluation; TUBerlin dataset; Pretrained Conv on Quickdraw. RNN + CNN Late Fusion"

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
    python tuberlin_fusion_eval.py \
        --checkpoint "/host${CL_LOG_DIR}${CL_RUNNAME}/${CL_CKPT_PREFIX}" \
        --dataset_fn ${CL_DATASET} \
        --dataset_root "/host${CL_DATASET_ROOT}" \
        --fusion ${CL_FUSION} \
        --intensity_channels ${CL_INTENSITY_CHANNELS} \
        --log_dir "/host${CL_LOG_DIR}${CL_RUNNAME}_eval" \
        --model_fn "${CL_MODEL}" \
        --note "${CL_NOTE}" \
    2>&1 | tee -a "${CL_LOG_DIR}${CL_RUNNAME}_eval/eval.log"
