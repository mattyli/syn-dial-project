ORIG_DATASET_FILENAME="./data/datasets/seed.jsonl"
NEW_DATASET_FILENAME="./data/generated/new-batch-1.solutions.jsonl"
SAVE_FILENAME="./data/complete/new-batch-1.jsonl"

GRADIENT_DIR="./data/gradient_storage/train"

cd "$(realpath ..)"
python cluster_filter.py --orig_dataset_filename=$ORIG_DATASET_FILENAME --new_dataset_filename=$NEW_DATASET_FILENAME \
  --save_filename=$SAVE_FILENAME --gradient_dir=$GRADIENT_DIR
