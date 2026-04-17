MODEL_NAME="Qwen2.5-72B"

INPUT_FILENAME="./data/generated/new-batch-1.problems.jsonl"
OUTPUT_FILENAME="./data/generated/new-batch-1.solutions.jsonl"

cd "$(realpath ..)"
python generate_solution.py --model_name=$MODEL_NAME --input_filename=$INPUT_FILENAME --output_filename=$OUTPUT_FILENAME
