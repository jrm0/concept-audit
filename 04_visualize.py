"""
04_visualize.py — Generate visualizations for feasibility test results.

Usage:
    python 04_visualize.py

Reads:
    data/embeddings_{concept}.npz
    results/metrics_summary.csv
    results/metrics_detail.json

Outputs:
    results/fig_css_comparison.png — Bar chart comparing CSS scores
    results/fig_mev_comparison.png — Bar chart comparing MEV scores
    results/fig_semantic_clouds.png — PCA projections of all concept clouds
    results/fig_cloud_detail_{concept}.png — Per-concept cloud with GMM clusters
"""

import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

CONCEPTS = ["reaction_time", "working_memory", "well_being", "resilience", "self_esteem"]

CONCEPT_LABELS = {
    "reaction_time": "Reaction Time",
    "working_memory": "Working Memory",
    "well_being": "Well-Being",
    "resilience": "Resilience",
    "self_esteem": "Self-Esteem",
}

TIER_COLORS = {
    "reaction_time": "#2ecc71",      # green (stable)
    "working_memory": "#27ae60",     # green (stable)
    "well_being": "#e74c3c",         # red (unstable)
    "resilience": "#c0392b",         # red (unstable)
    "self_esteem": "#3498db",        # blue (synonym)
}


def load_metrics():
    """Load the detailed metrics JSON."""
    with open(RESULTS_DIR / "metrics_detail.json") as f:
        return json.load(f)


def plot_css_comparison(metrics):
    """Bar chart comparing CSS scores across concepts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    concepts = [m["concept"] for m in metrics]
    scores = [m["css_score"] for m in metrics]
    colors = [TIER_COLORS.get(c, "#95a5a6") for c in concepts]
    labels = [CONCEPT_LABELS.get(c, c) for c in concepts]

    # Sort by score
    sorted_idx = np.argsort(scores)
    labels = [labels[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Conceptual Stability Score (0-100)", fontsize=12)
    ax.set_title("Conceptual Stability Score: Feasibility Test", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va="center", fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Expected Stable"),
        Patch(facecolor="#e74c3c", label="Expected Unstable"),
        Patch(facecolor="#3498db", label="Synonym Candidate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_css_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_css_comparison.png")


def plot_mev_comparison(metrics):
    """Bar chart comparing MEV (Maximum Explained Variance) across concepts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    concepts = [m["concept"] for m in metrics]
    mevs = [m["mev"]["mev"] for m in metrics]
    colors = [TIER_COLORS.get(c, "#95a5a6") for c in concepts]
    labels = [CONCEPT_LABELS.get(c, c) for c in concepts]

    sorted_idx = np.argsort(mevs)[::-1]
    labels = [labels[i] for i in sorted_idx]
    mevs = [mevs[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(labels, mevs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Maximum Explained Variance (PC1)", fontsize=12)
    ax.set_title("MEV: How Well Can a Single Vector Represent This Concept?", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)

    for bar, mev in zip(bars, mevs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{mev:.3f}", va="center", fontsize=11)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_mev_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_mev_comparison.png")


def plot_semantic_clouds(concepts_list):
    """PCA projection of all concept clouds, overlaid."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Load all embeddings and fit a shared PCA
    all_embs = []
    all_labels = []
    for concept in concepts_list:
        path = DATA_DIR / f"embeddings_{concept}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        embs = data["embeddings"]
        # Subsample for visualization
        if len(embs) > 200:
            idx = np.random.choice(len(embs), 200, replace=False)
            embs = embs[idx]
        all_embs.append(embs)
        all_labels.extend([concept] * len(embs))

    if not all_embs:
        print("No embeddings found for visualization")
        return

    combined = np.vstack(all_embs)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    # Plot each concept
    offset = 0
    for embs, concept in zip(all_embs, concepts_list):
        n = len(embs)
        pts = projected[offset:offset + n]
        color = TIER_COLORS.get(concept, "#95a5a6")
        label = CONCEPT_LABELS.get(concept, concept)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.3, s=15, label=label)
        # Mark centroid
        centroid = pts.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=color, s=150, marker="X",
                   edgecolors="black", linewidths=1, zorder=5)
        offset += n

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("Semantic Clouds: All Concepts (PCA Projection)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, markerscale=2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_semantic_clouds.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_semantic_clouds.png")


def plot_cloud_detail(concept_slug, metrics):
    """Per-concept cloud with GMM clusters overlaid."""
    path = DATA_DIR / f"embeddings_{concept_slug}.npz"
    if not path.exists():
        return

    data = np.load(path, allow_pickle=True)
    embs = data["embeddings"]

    # Find this concept's metrics
    concept_metrics = next((m for m in metrics if m["concept"] == concept_slug), None)
    best_k = concept_metrics["clustering"]["best_k"] if concept_metrics else 2

    # PCA to 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(10, 8))

    if best_k > 1:
        # Fit GMM on PCA-reduced data
        pca_for_gmm = PCA(n_components=min(5, len(embs) - 1))
        reduced = pca_for_gmm.fit_transform(embs)
        gmm = GaussianMixture(n_components=best_k, n_init=50, random_state=42)
        labels = gmm.fit_predict(reduced)

        scatter = ax.scatter(projected[:, 0], projected[:, 1],
                             c=labels, cmap="Set2", alpha=0.5, s=20)
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        color = TIER_COLORS.get(concept_slug, "#95a5a6")
        ax.scatter(projected[:, 0], projected[:, 1], c=color, alpha=0.4, s=20)

    label = CONCEPT_LABELS.get(concept_slug, concept_slug)
    css = concept_metrics["css_score"] if concept_metrics else "?"
    mev = concept_metrics["mev"]["mev"] if concept_metrics else "?"

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title(
        f"Semantic Cloud: {label}\n"
        f"CSS={css:.1f}  |  MEV={mev:.3f}  |  Clusters={best_k}",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"fig_cloud_detail_{concept_slug}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved fig_cloud_detail_{concept_slug}.png")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()

    print("Generating visualizations...\n")
    plot_css_comparison(metrics)
    plot_mev_comparison(metrics)
    plot_semantic_clouds(CONCEPTS)

    for concept in CONCEPTS:
        plot_cloud_detail(concept, metrics)

    print("\n=== Visualization complete ===")
    print(f"All figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
