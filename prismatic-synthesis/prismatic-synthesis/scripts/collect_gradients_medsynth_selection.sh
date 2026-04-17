MODEL_NAME="/model-weights/Qwen3-0.6B"
DATASET_FILENAME="../../g-vendi/data/datasets/medsynth_selection.jsonl"
SAVE_DIR="../../g-vendi/data/gradient_storage/medsynth-qwen3-0.6b"
SAVE_FILE_PREFIX="medsynth"
NUM_GPUS=8

cd "$(realpath ..)"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python collect_gradients.py \
    --model_name_or_path=$MODEL_NAME \
    --dataset_filename=$DATASET_FILENAME \
    --save_dir=$SAVE_DIR \
    --save_file_prefix=$SAVE_FILE_PREFIX \
    --device_split_size=$NUM_GPUS &
done
wait
