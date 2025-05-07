import re
import numpy as np
import torch
import torch.nn.functional as F
from wtpsplit import SaT
from huggingface_hub import login
from transformers import DPRQuestionEncoder, DPRContextEncoder


#hugging face-cli login --token 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
# Load BhLegalBERT dual encoders (DPR style)

login(token="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")
query_encoder = DPRQuestionEncoder.from_pretrained("RvShivam/BhLegalBERT", subfolder="retrieval")

context_encoder = DPRContextEncoder.from_pretrained("RvShivam/BhLegalBERT", subfolder="retrieval")

def split_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences using wtpsplit.
    """
    sat = SaT("sat-3l")  
    return sat.split(text)


def encode_query(query: str, device: str = "cpu") -> torch.Tensor:
    """
    Encode the query string into a dense vector.
    """
    inputs = query_encoder.tokenizer([query], return_tensors="pt")
    outputs = query_encoder(**{k: v.to(device) for k, v in inputs.items()})
    # Return single vector of shape (hidden_size,)
    return outputs.pooler_output[0]

def encode_sentences(sentences: list[str], device: str = "cpu") -> torch.Tensor:
    """
    Encode a list of sentences into dense vectors.
    Returns a tensor of shape (num_sentences, hidden_size).
    """
    inputs = context_encoder.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    outputs = context_encoder(**{k: v.to(device) for k, v in inputs.items()})
    return outputs.pooler_output  # shape: (n, hidden_size)

def coarse_ranking(query_emb: torch.Tensor, sent_embs: torch.Tensor, top_m: int = 50) -> list[int]:
    """
    Perform coarse ranking of sentences via cosine similarity.
    Returns indices of top_m most similar sentences.
    """
    sims = F.cosine_similarity(sent_embs, query_emb.unsqueeze(0), dim=1)
    topk = sims.topk(min(top_m, len(sims))).indices.tolist()
    return topk

def mmr(sent_embs: np.ndarray, query_emb: np.ndarray, top_n: int, lambda_param: float = 0.7) -> list[int]:
    """
    Apply Maximal Marginal Relevance to select top_n diverse sentences.
    Returns list of selected indices.
    """
    selected = []
    unselected = list(range(len(sent_embs)))

    # Precompute similarity to query
    sim_to_query = np.dot(sent_embs, query_emb) / (
        np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(query_emb)
    )
    # Precompute pairwise similarity matrix
    normed = sent_embs / np.linalg.norm(sent_embs, axis=1, keepdims=True)
    sim_matrix = np.dot(normed, normed.T)

    for _ in range(min(top_n, len(unselected))):
        mmr_scores = {}
        for i in unselected:
            relevance = sim_to_query[i]
            redundancy = max([sim_matrix[i][j] for j in selected] or [0])
            mmr_scores[i] = lambda_param * relevance - (1 - lambda_param) * redundancy
        best = max(mmr_scores, key=mmr_scores.get)
        selected.append(best)
        unselected.remove(best)
    return selected

def enforce_clauses(sentences: list[str], selected_idxs: list[int]) -> list[int]:
    """
    Ensure that sentences containing legal markers are included.
    Adds indices for sentences with 'Section' or 'hereby'.
    """
    for idx, sent in enumerate(sentences):
        if re.search(r"\bSection\s*\d+(?:\.\d+)*\b", sent) or re.search(r"\bhereby\b", sent, re.IGNORECASE):
            if idx not in selected_idxs:
                selected_idxs.append(idx)
    return selected_idxs

def select_sentences(chunk: str, device: str = "cpu", top_k: int = 10, top_m: int = 50, lambda_param: float = 0.7) -> list[str]:
    """
    Main extraction function:
      1. Split into sentences
      2. Encode sentences & query
      3. Coarse ranking to top_m
      4. MMR re-ranking to top_k
      5. Enforce legal clause inclusion
    Returns list of selected sentence texts.
    """
    sentences = split_sentences(chunk)
    if not sentences:
        return []

    # Encode
    sent_embs = encode_sentences(sentences, device).cpu().numpy()
    query_emb = encode_query("Summarize key legal points", device).cpu().numpy()

    # Coarse ranking
    top_m_idxs = coarse_ranking(torch.tensor(query_emb), torch.tensor(sent_embs), top_m)

    # MMR re-ranking
    mmr_idxs = mmr(sent_embs[top_m_idxs], query_emb, top_k, lambda_param)
    selected = [top_m_idxs[i] for i in mmr_idxs]

    # Enforce clauses
    final_idxs = enforce_clauses(sentences, selected)

    # Return selected sentences
    return [sentences[i] for i in sorted(final_idxs)]
