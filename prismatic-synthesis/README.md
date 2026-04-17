# Prismatic Synthesis
Supplementary code for `Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning`

## Installation
```bash
pip install -r requirements.txt
```

## G-Vendi Score
G-Vendi is a diversity metric that leverages gradients from off-the-shelf proxy model. Measuring G-Vendi consists of two steps:
1. Collect gradients using a small, off-the-shelf instruction-tuned model
2. Measure the exponeniated entropy of the samples in the gradient space.

### Step 1. Collect gradients
We first collect gradients using proxy model. We prepared a sample dataset as shown in `g-vendi/data/datasets`.

Our logic for gradient collection is implemented in `g-vendi/gradient_computer.py` and `g-vendi/collect_gradients.py`. To run the logic, simply execute
```bash
cd g-vendi/scripts
bash collect_gradients.sh
```

This will create corresponding gradient files in `.safetensors` format at `g-vendi/data/gradient_storage/`.

### Step 2. Measure gradient entropy
We compute the exponentiated entropy of the collected gradients as our diversity metric. 

The corresponding codes are in `g-vendi/gradient_vendi.py` and `g-vendi/compute_g-vendi.py`. To run the code for our sample dataset, run
```bash
python compute_g-vendi.py --dataset_filename=./data/datasets/seed.jsonl --gradient_storage=./data/gradient_storage/train--qwen2.5-0.5b-instruct
```

## Prismatic Synthesis
We also present Prismatic Synthesis, an algorithm to generate novel synthetic data while improving overall diversity.

All our logic for Prismatic Synthesis is implemented in `prismatic-synthesis/*_modules`. We also prepare script files to illustrate single iteration loop of Prismatic Synthesis as below.

### Step 1. Collect seed set gradients
First, collect gradients for the seed set by running:
```bash
cd prismatic-synthesis/scripts
bash collect_seed_set_gradients.sh
```
This will collect gradients of the seed samples in `.safetensors` format at `prismatic-synthesis/data/gradient_storage/`.

### Step 2. Generate problems
Next, generate problems and corresponding solutions from seed samples.
```bash
bash generate_problem.sh
bash generate_solution.sh
```
This will generate novel samples in `prismatic-synthesis/data/generated/`.

### Step 3. Compute gradients for new samples
Compute gradients for the generated samples by
```bash
bash collect_new_gradient.sh
```

### Step 4. Rejection-sample only the sparse clusters
Finally, filter out samples that correspond to already popular clusters.
```bash
bash cluster_filter.sh
```
This will leave only the novel, under-represented samples in `prismatic-synthesis/data/complete/new-batch-1.jsonl`.
