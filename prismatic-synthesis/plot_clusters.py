"""
Regenerate the K-means cluster plot from a previous select_diverse_subset.py run.

Skips gradient collection and model inference entirely — only needs the saved
gradient files and the outputs already written to --output_dir.

Usage:
    python prismatic-synthesis/plot_clusters.py \
        --dataset_filename prismatic-synthesis/g-vendi/data/datasets/medsynth_selection.jsonl \
        --gradient_dir prismatic-synthesis/g-vendi/data/gradient_storage/medsynth-qwen3-0.6b \
        --subset_jsonl prismatic-synthesis/results/selected_subset_N1000.jsonl \
        --output_path prismatic-synthesis/results/cluster_plot_N1000.png
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

_HERE = Path(__file__).resolve().parent / "prismatic-synthesis"
sys.path.insert(0, str(_HERE))

from cluster_modules.cluster_manager import ClusterManager
from gradient_modules.gradient_manager import GradientManager


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--gradient_dir", type=str, required=True)
    parser.add_argument("--subset_jsonl", type=str, required=True,
                        help="Path to selected_subset_N*.jsonl from a previous run")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to write the PNG (e.g. results/cluster_plot.png)")
    args = parser.parse_args()
    args.dataset_filename = Path(args.dataset_filename)
    args.gradient_dir = Path(args.gradient_dir)
    args.subset_jsonl = Path(args.subset_jsonl)
    args.output_path = Path(args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()

    # Load full dataset and selected subset to recover selected indices
    with jsonlines.open(args.dataset_filename) as f:
        samples = list(f)
    sample_ids = [s["id"] for s in samples]

    with jsonlines.open(args.subset_jsonl) as f:
        selected_ids = {s["id"] for s in f}

    selected_indices = [i for i, sid in enumerate(sample_ids) if sid in selected_ids]
    print(f"Matched {len(selected_indices)} selected examples from {args.subset_jsonl.name}")

    # Load all available gradients then filter to dataset order
    all_ids, all_gradients = GradientManager.load_all_gradients(args.gradient_dir)
    grad_lookup = {sid: all_gradients[i] for i, sid in enumerate(all_ids)}
    loaded_ids = [sid for sid in sample_ids if sid in grad_lookup]
    gradients = torch.stack([grad_lookup[sid] for sid in loaded_ids])
    print(f"Loaded gradients for {len(loaded_ids)} / {len(sample_ids)} examples  shape={tuple(gradients.shape)}")

    # Remap selected_indices to the loaded_ids index space
    loaded_id_set = {sid: i for i, sid in enumerate(loaded_ids)}
    selected_indices = [loaded_id_set[sample_ids[i]] for i in selected_indices if sample_ids[i] in loaded_id_set]

    # Re-run K-means with same settings (deterministic: init = first k points)
    k = max(1, len(loaded_ids) // 10)
    print(f"Running K-means  k={k}  iters=20 ...")
    labels, centroids = ClusterManager.cluster_kmeans(
        gradients.cuda(), k=k, num_iter=20, use_tqdm=True
    )

    # Plot
    k_actual = centroids.size(0)
    grad_np = F.normalize(gradients, dim=1).cpu().numpy()
    centroid_np = centroids.cpu().numpy()
    labels_np = labels.cpu().numpy()

    all_points = np.concatenate([grad_np, centroid_np], axis=0)
    pca = PCA(n_components=2, random_state=0)
    all_2d = pca.fit_transform(all_points)
    data_2d = all_2d[: len(grad_np)]
    centroid_2d = all_2d[len(grad_np):]

    cmap = plt.get_cmap("tab20" if k_actual <= 20 else "hsv")
    colors = [cmap(i / k_actual) for i in labels_np]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, s=6, alpha=0.4, linewidths=0)

    # sel = np.array(sorted(selected_indices))
    # ax.scatter(
    #     data_2d[sel, 0], data_2d[sel, 1],
    #     s=20, facecolors="none", edgecolors="black", linewidths=0.6, label="selected",
    # )
    # ax.scatter(
    #     centroid_2d[:, 0], centroid_2d[:, 1],
    #     marker="x", s=60, c="red", linewidths=1.2, label="centroids",
    # )

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title(f"K-means clusters (k={k_actual}, n={len(grad_np)})")
    ax.legend(markerscale=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote cluster plot → {args.output_path}")


if __name__ == "__main__":
    main()
