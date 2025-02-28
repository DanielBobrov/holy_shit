import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: List[str],
    mask_position: int,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
):
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape [seq_len, seq_len]
        tokens: List of token strings
        mask_position: Position of the mask token
        title: Title for the plot
        save_path: Path to save the figure (if None, display instead)
    """
    plt.figure(figsize=(10, 8))
    
    # Extract attention from mask token to all other tokens
    mask_attention = attention_weights[mask_position].cpu().numpy()
    
    # Plot heatmap
    sns.heatmap(
        mask_attention.reshape(1, -1),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=tokens,
        yticklabels=["Mask"],
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_representation_trajectory(
    representations: List[torch.Tensor],
    method: str = "pca",
    title: str = "Representation Evolution",
    save_path: Optional[str] = None
):
    """
    Plot the trajectory of a representation across iterations.
    
    Args:
        representations: List of representation tensors
        method: Dimensionality reduction method ('pca' or 'tsne')
        title: Title for the plot
        save_path: Path to save the figure (if None, display instead)
    """
    # Stack representations
    stacked_reps = torch.stack(representations).cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=5)
    
    reduced_reps = reducer.fit_transform(stacked_reps.reshape(len(representations), -1))
    
    # Plot trajectory
    plt.figure(figsize=(10, 8))
    
    plt.plot(
        reduced_reps[:, 0],
        reduced_reps[:, 1],
        marker='o',
        linestyle='-',
        markersize=8
    )
    
    # Add arrows to show direction
    for i in range(len(reduced_reps) - 1):
        plt.arrow(
            reduced_reps[i, 0],
            reduced_reps[i, 1],
            (reduced_reps[i+1, 0] - reduced_reps[i, 0]) * 0.9,
            (reduced_reps[i+1, 1] - reduced_reps[i, 1]) * 0.9,
            head_width=0.1,
            head_length=0.1,
            fc='black',
            ec='black'
        )
    
    # Add iteration labels
    for i, (x, y) in enumerate(reduced_reps):
        plt.text(x, y, f"Iter {i}", fontsize=12, ha='right', va='bottom')
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def visualize_token_prediction_confidence(
    logits_by_iteration: List[torch.Tensor],
    mask_position: int,
    top_k: int = 5,
    token_map: Dict[int, str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize how prediction confidence for top tokens changes across iterations.
    
    Args:
        logits_by_iteration: List of logits tensors [batch_size, seq_len, vocab_size]
        mask_position: Position of the mask token
        top_k: Number of top tokens to show
        token_map: Mapping from token IDs to readable names
        save_path: Path to save the figure (if None, display instead)
    """
    plt.figure(figsize=(12, 8))
    
    # Extract logits at mask position for each iteration
    mask_logits = [logits[0, mask_position] for logits in logits_by_iteration]
    
    # Get the top-k token IDs from the final iteration
    final_probs = F.softmax(mask_logits[-1], dim=-1)
    top_token_ids = torch.topk(final_probs, top_k).indices.cpu().numpy()
    
    # Track probabilities of these tokens across iterations
    iterations = list(range(len(mask_logits)))
    for token_id in top_token_ids:
        token_probs = [F.softmax(logits, dim=-1)[token_id].item() for logits in mask_logits]
        
        token_name = f"Token {token_id}"
        if token_map and token_id in token_map:
            token_name = token_map[token_id]
            
        plt.plot(iterations, token_probs, marker='o', linewidth=2, label=token_name)
    
    plt.title("Prediction Confidence Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
