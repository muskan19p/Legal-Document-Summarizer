import numpy as np
from typing import List


def compute_cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row in matrix and the vector.

    Args:
        matrix: np.ndarray of shape (n_items, dim)
        vector: np.ndarray of shape (dim,)

    Returns:
        similarities: np.ndarray of shape (n_items,)
    """
    norm_matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_vector = vector / np.linalg.norm(vector)
    return np.dot(norm_matrix, norm_vector)


def mmr_selection(
    candidate_embs: np.ndarray,
    query_emb: np.ndarray,
    top_k: int,
    lambda_param: float = 0.7
) -> List[int]:
    """
    Selects indices of top_k items using Maximal Marginal Relevance (MMR).

    Args:
        candidate_embs: np.ndarray of shape (n_candidates, dim)
        query_emb: np.ndarray of shape (dim,)
        top_k: number of items to select
        lambda_param: trade-off parameter between relevance and diversity

    Returns:
        selected_indices: List[int]
    """
    n_candidates = candidate_embs.shape[0]
    # Compute relevance scores
    relevance = compute_cosine_similarity(candidate_embs, query_emb)
    # Compute pairwise similarities
    normed = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    sim_matrix = np.dot(normed, normed.T)

    selected = []
    unselected = list(range(n_candidates))

    for _ in range(min(top_k, n_candidates)):
        best_idx = None
        best_score = -np.inf
        for idx in unselected:
            # Compute redundancy
            max_sim = max(sim_matrix[idx][j] for j in selected) if selected else 0
            # MMR score: relevance - diversity penalty
            score = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        unselected.remove(best_idx)

    return selected
