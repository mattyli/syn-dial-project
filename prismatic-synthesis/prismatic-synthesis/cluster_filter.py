import os
import sys
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List

import ipdb
import jsonlines
import torch.nn.functional as F
import torch

from cluster_modules.cluster_manager import ClusterManager
from generation_modules.generate_model_util import save_to_file
from gradient_modules.gradient_manager import GradientManager


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--orig_dataset_filename", type=str, required=True)
    parser.add_argument("--new_dataset_filename", type=str, required=True)
    parser.add_argument("--save_filename", type=str, required=True)

    parser.add_argument("--gradient_dir", type=str, required=True)

    args = parser.parse_args()

    args.orig_dataset_filename = Path(args.orig_dataset_filename)
    args.new_dataset_filename = Path(args.new_dataset_filename)

    args.save_filename = Path(args.save_filename)
    os.makedirs(args.save_filename.parent, exist_ok=True)
    assert not args.save_filename.exists(), f"{args.save_filename} already exists."

    args.gradient_dir = Path(args.gradient_dir)

    return args


def filter_cluster(
        sample_gradients: torch.Tensor, ratio: float,
        current_cluster_centroids: torch.Tensor, current_cluster_labels: torch.Tensor,
) -> List[bool]:
    # -- filter only the samples whose gradient corresponds to the smallest N% clusters -- #
    # normalize gradients
    sample_gradients = F.normalize(sample_gradients, dim=1)

    # assign cluster labels to samples
    similarity_with_centroids = sample_gradients @ current_cluster_centroids.T  # (# samples, # centroids)
    sample_cluster_labels = torch.argmax(similarity_with_centroids, dim=-1).tolist()  # (# samples,)

    # compute the smallest `ratio` clusters
    small_clusters = ClusterManager.smallest_clusters(current_cluster_labels, ratio)

    return [label in small_clusters for label in sample_cluster_labels]


if __name__ == "__main__":
    args = parse_args()

    # -- Load gradients for current data pool -- #
    with jsonlines.open(args.orig_dataset_filename, "r") as f:
        orig_sample_ids = [s['id'] for s in list(f)]

    _, orig_sample_gradients = GradientManager.load_gradients_for_sample_ids(args.gradient_dir, orig_sample_ids)

    # -- Run K-means with current data pool -- #
    cluster_labels, cluster_centroids = ClusterManager.cluster_kmeans(
        orig_sample_gradients, k=int(orig_sample_gradients.size(0) * 0.1), num_iter=20, use_tqdm=True
    )

    with jsonlines.open(args.new_dataset_filename, "r") as f:
        new_samples = list(f)
        new_sample_ids = [s['id'] for s in new_samples]

    _, new_sample_gradients = GradientManager.load_gradients_for_sample_ids(args.gradient_dir, new_sample_ids)

    # -- Cluster-based Filtering: leave only the samples whose gradient corresponds to the small clusters -- #
    filter_cluster_results = filter_cluster(
        new_sample_gradients, ratio=0.5,
        current_cluster_centroids=cluster_centroids, current_cluster_labels=cluster_labels
    )

    # -- Add those samples that pass filtering stage -- #
    out_samples = []
    for filter_result, new_sample in zip(filter_cluster_results, new_samples):
        if filter_result:
            out_samples.append(new_sample)

    save_to_file(out_samples, args.save_filename)
