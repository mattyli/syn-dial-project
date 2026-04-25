# Diversity Selection Pipeline Spec

## Overview

Selects a diverse subset of N dialogue→note pairs from MedSynth using gradient-based clustering. The core idea: represent each training example by the gradient it induces on a small proxy model, then cluster those gradients and sample uniformly across clusters — ensuring the selected subset covers the full diversity of the dataset rather than overrepresenting frequent patterns.

The pipeline has three stages that run sequentially:

```
MedSynth_huggingface_final.csv
        │
        ▼ [Stage 1] prepare_medsynth_data.py (local, CPU)
medsynth_selection.jsonl  (10,033 examples)
        │
        ▼ [Stage 2] collect_gradients.py (Slurm array, 8 × l40s GPU)
gradient_storage/medsynth_selection--qwen3-0.6b/  (safetensors, 1024-D per example)
        │
        ▼ [Stage 3] select_diverse_subset.py (Slurm, 1 × l40s GPU)
selected_subset_N{N}_K{k}.jsonl
selected_subset_N{N}_K{k}_metadata.json
cluster_plot_N{N}_K{k}.png
```

---

## Stage 1: Data Preparation

**Script:** `prismatic-synthesis/prepare_medsynth_data.py`  
**Runs:** locally (no GPU required)

Reads `MedSynth_huggingface_final.csv` (10,033 rows, columns `Dialogue` and ` Note`) and writes three JSONL files with `{"id", "prompt", "completion"}` records:

| Output file | Prompt | Completion | Purpose |
|---|---|---|---|
| `medsynth_notes.jsonl` | dialogue | note | measure note diversity |
| `medsynth_dialogues.jsonl` | note | dialogue | measure dialogue diversity |
| `medsynth_selection.jsonl` | normalized dialogue | note | **used for gradient collection** |

Speaker normalization in `medsynth_selection.jsonl`: `[doctor]` → `Doctor:`, `[patient]` → `Patient:` (case-insensitive, strips trailing colon if present). IDs are `note_<row_index>`.

---

## Stage 2: Gradient Collection

**Script:** `prismatic-synthesis/g-vendi/collect_gradients.py`  
**Runs:** Slurm array job, 8 tasks × 1 l40s GPU each  
**Proxy model:** `/model-weights/Qwen3-0.6B` (loaded at full precision, eval mode)

### Sharding

The 10,033-example dataset is split into 8 contiguous shards. Each Slurm task is identified by `CUDA_VISIBLE_DEVICES` (0–7) and processes `ceil(n / 8)` examples. Checkpointing: on restart, the script scans existing `.txt` files in the output directory to find the last processed index for its shard and resumes from there.

### Gradient computation (`GradientComputer`)

For each example `{id, prompt, completion}`:

1. **Tokenize** using Qwen chat template with `DataCollatorForCompletionOnlyLM` to mask the prompt in `labels` (loss computed on completion tokens only).
2. **Forward + backward pass** on the frozen proxy model.
3. **Vectorize** all parameter gradients: `cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])` → full gradient vector of dimension `grad_dim` (≈ all parameters of Qwen3-0.6B).
4. **Accumulate** into a batch of 4 examples (`project_interval=4`), then **project** to 1024-D using a Rademacher random projection (TRAK `CudaProjector` / `BasicProjector` fallback):

   ```
   projected_gradient = projector.project(full_grads, model_id=0) / sqrt(proj_dim)
   ```

   Projection dtype: float16. This is a Johnson–Lindenstrauss projection that preserves inner products: `g_i · g_j ≈ projected_g_i · projected_g_j`.

5. **Save** every 500 examples to a safetensors file named `<start_index>.safetensors` + companion `<start_index>.txt` (JSONL of sample IDs for that shard file).

### Output layout

```
gradient_storage/medsynth_selection--qwen3-0.6b/
    0.safetensors        # gradients for examples 0–499 on GPU 0
    0.txt                # sample IDs for that file
    500.safetensors
    500.txt
    ...                  # one pair per 500-example batch, across all 8 GPU shards
```

Each safetensors file maps `sample_id → float16 tensor of shape (1024,)`.

---

## Stage 3: Subset Selection

**Script:** `prismatic-synthesis/select_diverse_subset.py`  
**Runs:** Slurm job (1 × l40s GPU), depends on all Stage 2 tasks completing  
**Parameters:** `--subset_size N`, `--k_clusters K` (default: `n // 10`)

### Step-by-step

#### 3a. Load gradients

`GradientManager.load_gradients_for_sample_ids` scans all `*.safetensors` files in the gradient directory, loads only the IDs matching `medsynth_selection.jsonl`, and stacks them into a single tensor of shape `(n, 1024)`.

#### 3b. K-means clustering (`ClusterManager.cluster_kmeans`)

Clusters the gradient vectors using spherical K-means (cosine similarity) on GPU:

1. **L2-normalize** all gradient vectors: `data = F.normalize(data, dim=1)`.
2. **Initialize centroids** as the first `k` rows of the normalized data (deterministic).
3. **Lloyd's algorithm**, 20 iterations:
   - **E-step (assignment):** For each point, find the nearest centroid by cosine similarity. Computed in batches of 90 centroids at a time to fit in VRAM:
     ```
     similarity = data @ centroids[batch].T   # (n, batch_k)
     labels = argmax over all batches
     ```
   - **M-step (update):** For each cluster `c`, set `centroids[c] = sum(data[labels == c])`, then L2-normalize.
4. Returns `labels` (n,) and `centroids` (k, 1024).

**Experimental configuration:** K=100 clusters across all runs (N ∈ {1000, 2500, 5000, 7500}).

#### 3c. Inverse-size weighting and sampling

```python
cluster_sizes = bincount(labels)               # (k,)  how many examples per cluster
weights = 1.0 / cluster_sizes[labels]          # (n,)  each example's weight
weights = weights / weights.sum()              # normalize to probability distribution
selected_indices = multinomial(weights, N, replacement=False)
```

This makes examples in smaller clusters more likely to be selected, spreading the sample evenly across the gradient space rather than concentrating it in high-density regions.

#### 3d. Outputs

Selected examples are written in **sorted index order** (reproducible):

- `selected_subset_N{N}_K{k}.jsonl` — JSONL records from the original dataset
- `selected_subset_N{N}_K{k}_metadata.json` — run provenance: proxy model, source paths, cluster size statistics, timestamp
- `cluster_plot_N{N}_K{k}.png` — 2D PCA projection of all gradient vectors, colored by cluster, with selected examples outlined in black and centroids marked as red ×

---

## Key design choices

| Choice | Rationale |
|---|---|
| Gradient-based representation | Gradients encode how each example changes model parameters, capturing semantic + structural difficulty rather than surface similarity |
| Qwen3-0.6B as proxy | Small enough to run gradient collection across 10K examples in reasonable time; gradients in the projected space are still informative for clustering |
| Rademacher JL projection to 1024-D | Reduces memory and clustering cost while preserving inner products (dot-product ≈ cosine similarity after normalization) |
| Cosine K-means (spherical) | Direction rather than magnitude matters for gradient similarity; magnitude varies by example length |
| Inverse cluster-size weights | Prevents large clusters (common patterns) from dominating the sample; ensures rare clinical scenarios are represented |
| K=100 clusters | Chosen to balance granularity against noise; with n≈10K examples, mean cluster size ≈ 100 |

---

## Reproducing an existing run

```bash
# Activate environment
source ~/projects/aip-zhu2048/mattli/syn-dial-project/.venv/bin/activate
export HF_HOME="$HOME/projects/aip-zhu2048/mattli/hf_cache"

# Gradients are already collected; run selection only:
python prismatic-synthesis/select_diverse_subset.py \
    --dataset_filename prismatic-synthesis/g-vendi/data/datasets/medsynth_selection.jsonl \
    --gradient_dir prismatic-synthesis/g-vendi/data/gradient_storage/medsynth_selection--qwen3-0.6b \
    --subset_size 1000 \
    --output_dir prismatic-synthesis/results/ \
    --k_clusters 100
```

To re-run the full pipeline from scratch (including gradient collection):

```bash
bash prismatic-synthesis/run_pipeline.sh 1000
```
