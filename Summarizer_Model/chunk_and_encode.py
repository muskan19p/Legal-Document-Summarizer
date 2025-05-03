from transformers import BigBirdTokenizerFast, BigBirdModel
import torch
import re

# 1. Initialize tokenizer and model
tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-base")
bigbird = BigBirdModel.from_pretrained("google/bigbird-roberta-base")

def chunk_text(text: str,
               max_length: int = 2048,
               overlap: int = 256) -> list[str]:
    """
    Split text into chunks by section headings or token windows.
    """
    # First try section-aware splitting
    sections = re.split(r'\n(?=\d+(\.\d+)*\s)', text)
    chunks = []
    for sec in sections:
        tokens = tokenizer(sec)["input_ids"]
        if len(tokens) <= max_length:
            chunks.append(sec)
        else:
            # fallback to sliding window
            for i in range(0, len(tokens), max_length - overlap):
                window_tokens = tokens[i:i + max_length]
                chunks.append(tokenizer.decode(window_tokens, skip_special_tokens=True))
    return chunks

def encode_chunks(chunks: list[str]) -> torch.Tensor:
    """
    Encode each text chunk into dense representations.
    Returns a tensor of shape (num_chunks, hidden_size).
    """
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", 
                           max_length=4096, truncation=True,
                           padding="max_length")
        with torch.no_grad():
            outputs = bigbird(**inputs)
            # Mean-pool over sequence dimension
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)  # shape: (num_chunks, hidden_size)
