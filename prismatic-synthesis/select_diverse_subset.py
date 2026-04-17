"""
Select a diverse subset of N dialogue→note examples from MedSynth using gradient-based clustering.

Usage:
    python prismatic-synthesis/select_diverse_subset.py \
        --dataset_filename prismatic-synthesis/g-vendi/data/datasets/medsynth_selection.jsonl \
        --gradient_dir prismatic-synthesis/g-vendi/data/gradient_storage/medsynth-qwen3-0.6b \
        --subset_size 1000 \
        --output_dir prismatic-synthesis/results/ \
        [--k_clusters 100] \
        [--no-plot]

Pipeline:
    1. Load per-example gradient vectors (1024-D, precomputed by collect_gradients.py)
    2. K-means cluster (k = --k_clusters or n // 10 by default, 20 Lloyd iterations, cosine similarity)
    3. Weight each example inversely proportional to its cluster size
    4. Sample N examples without replacement according to those weights
    5. Write selected_subset_N{N}_K{k}.jsonl + selected_subset_N{N}_K{k}_metadata.json
    6. (Optional) Plot 2D PCA projection of clusters → cluster_plot_N{N}_K{k}.png
"""

import json
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import jsonlines
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Add prismatic-synthesis/ to path so we can import its modules
_HERE = Path(__file__).resolve().parent / "prismatic-synthesis"
sys.path.insert(0, str(_HERE))

from cluster_modules.cluster_manager import ClusterManager
from gradient_modules.gradient_manager import GradientManager
from generation_modules.generate_model_util import save_to_file


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--gradient_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k_clusters", type=int, default=None, help="Number of K-means clusters (default: n // 10)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Skip the 2D PCA cluster visualization PNG")
    parser.set_defaults(plot=True)
    args = parser.parse_args()
    args.dataset_filename = Path(args.dataset_filename)
    args.gradient_dir = Path(args.gradient_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def plot_clusters(
    gradients: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    selected_indices: list,
    output_path: Path,
) -> None:
    """Project gradients to 2D via PCA and save a scatter plot colored by cluster."""
    k = centroids.size(0)
    grad_np = F.normalize(gradients, dim=1).cpu().numpy()
    centroid_np = centroids.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # PCA on data + centroids together so they share the same embedding
    all_points = np.concatenate([grad_np, centroid_np], axis=0)
    pca = PCA(n_components=2, random_state=0)
    all_2d = pca.fit_transform(all_points)
    data_2d = all_2d[: len(grad_np)]
    centroid_2d = all_2d[len(grad_np):]

    # Use a cyclic colormap so adjacent cluster colors don't clash too badly
    cmap = plt.get_cmap("tab20" if k <= 20 else "hsv")
    colors = [cmap(i / k) for i in labels_np]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, s=6, alpha=0.4, linewidths=0)

    # Highlight selected examples
    sel = np.array(sorted(selected_indices))
    ax.scatter(
        data_2d[sel, 0], data_2d[sel, 1],
        s=20, facecolors="none", edgecolors="black", linewidths=0.6, label="selected",
    )

    # Mark centroids
    ax.scatter(
        centroid_2d[:, 0], centroid_2d[:, 1],
        marker="x", s=60, c="red", linewidths=1.2, label="centroids",
    )

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title(f"K-means clusters (k={k}, n={len(grad_np)})")
    ax.legend(markerscale=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote cluster plot → {output_path}")


def main():
    args = parse_args()
    N = args.subset_size

    # 1. Load dataset
    with jsonlines.open(args.dataset_filename) as f:
        samples = list(f)
    sample_ids = [s["id"] for s in samples]
    total = len(samples)
    print(f"Loaded {total} candidates from {args.dataset_filename}")

    # 2. Load gradients
    loaded_ids, gradients = GradientManager.load_gradients_for_sample_ids(
        args.gradient_dir, sample_ids
    )
    print(f"Loaded gradients for {len(loaded_ids)} / {total} examples  shape={tuple(gradients.shape)}")

    # 3. Cluster (k = --k_clusters if provided, else n // 10, 20 Lloyd iterations)
    k = args.k_clusters if args.k_clusters is not None else max(1, len(loaded_ids) // 10)
    print(f"Running K-means  k={k}  iters=20 ...")
    gradients_cuda = gradients.cuda()
    labels, centroids = ClusterManager.cluster_kmeans(gradients_cuda, k=k, num_iter=20, use_tqdm=True)

    cluster_sizes = torch.bincount(labels, minlength=k)  # (k,)
    nonzero = (cluster_sizes > 0).sum().item()
    sizes_sorted = cluster_sizes[cluster_sizes > 0].sort().values.tolist()
    print(f"Clusters formed: {nonzero}  |  size range [{sizes_sorted[0]}, {sizes_sorted[-1]}]  "
          f"mean={sum(sizes_sorted)/len(sizes_sorted):.1f}")

    # 4. Inverse-size weights → sample N without replacement
    per_sample_cluster = labels  # (n,)
    weights = 1.0 / cluster_sizes[per_sample_cluster].float()  # (n,)
    weights = F.normalize(weights.unsqueeze(0), p=1, dim=1).squeeze(0)  # normalize to sum=1

    actual_N = min(N, total)
    selected_indices = torch.multinomial(weights, actual_N, replacement=False).tolist()
    print(f"Selected {len(selected_indices)} examples (requested {N})")

    # 5. Build output records (preserving original order for reproducibility)
    selected_indices_set = set(selected_indices)
    selected_samples = [samples[i] for i in sorted(selected_indices_set)]

    # 6. Save subset JSONL
    out_jsonl = args.output_dir / f"selected_subset_N{actual_N}_K{k}.jsonl"
    save_to_file(selected_samples, out_jsonl)
    print(f"Wrote subset → {out_jsonl}")

    # 7. Save metadata
    meta = {
        "proxy_model": "/model-weights/Qwen3-0.6B",
        "source_dataset": str(args.dataset_filename),
        "gradient_dir": str(args.gradient_dir),
        "total_candidates": total,
        "gradients_loaded": len(loaded_ids),
        "subset_size": actual_N,
        "k_clusters": k,
        "clusters_nonempty": nonzero,
        "cluster_size_min": sizes_sorted[0],
        "cluster_size_max": sizes_sorted[-1],
        "cluster_size_mean": round(sum(sizes_sorted) / len(sizes_sorted), 2),
        "output_path": str(out_jsonl),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_meta = args.output_dir / f"selected_subset_N{actual_N}_K{k}_metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata → {out_meta}")

    # 8. Optional cluster plot
    if args.plot:
        plot_path = args.output_dir / f"cluster_plot_N{actual_N}_K{k}.png"
        plot_clusters(gradients, labels, centroids, selected_indices, plot_path)

    # 9. Summary
    print("\n=== Run Summary ===")
    print(f"  Candidates processed : {total}")
    print(f"  Gradients loaded     : {len(loaded_ids)}")
    print(f"  Clusters formed      : {nonzero} (k={k})")
    print(f"  Cluster size dist.   : min={sizes_sorted[0]}  max={sizes_sorted[-1]}  "
          f"mean={meta['cluster_size_mean']}")
    print(f"  Selected examples    : {len(selected_samples)}")


if __name__ == "__main__":
    main()
