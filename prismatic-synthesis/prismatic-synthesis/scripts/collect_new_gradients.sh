MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_FILENAME="./data/generated/new-batch-1.solutions.jsonl"
SAVE_DIR="./data/gradient_storage/train"
SAVE_FILE_PREFIX="new-batch-1"
NUM_GPUS=8

cd "$(realpath ..)"
for i in $(seq 0 "$((NUM_GPUS - 1))");
do
  CUDA_VISIBLE_DEVICES=$i python collect_gradients.py --model_name_or_path=$MODEL_NAME \
    --dataset_filename=$DATASET_FILENAME --save_dir=$SAVE_DIR \
    --save_file_prefix=$SAVE_FILE_PREFIX --device_split_size=$NUM_GPUS &
done
wait