"""
BERT embeddings.

The report uses TF Hub's BERT v4 encoder + v3 preprocessor; here we use the
HuggingFace `transformers` equivalent because it plays nicer with the rest
of the PyTorch pipeline (especially the FGSM gradient step which needs the
embeddings to be a leaf tensor in the computation graph).

We pool with CLS by default; mean-pooling is offered as an alternative
because on short noisy tweets it sometimes generalises better.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from .. import config


class BertEmbedder:
    """
    Generates a fixed-size embedding for each tweet.

    `pooling="cls"` returns the [CLS] token vector (768-d for bert-base).
    `pooling="mean"` averages across tokens with the attention mask applied.
    Cached embeddings can be saved to disk via `save_to_parquet()` so the
    expensive forward passes are only done once.
    """

    def __init__(self,
                 model_name: str | None = None,
                 max_len: int | None    = None,
                 pooling: str           = "cls",
                 device: str | None     = None):
        self.model_name = model_name or config.BERT_MODEL_NAME
        self.max_len    = max_len    or config.BERT_MAX_LEN
        self.pooling    = pooling
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    # ----------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(batch,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt").to(self.device)
            out = self.model(**tokens)

            if self.pooling == "cls":
                vecs = out.last_hidden_state[:, 0, :]
            else:
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)

            all_vectors.append(vecs.cpu().numpy())
        return np.vstack(all_vectors)

    # ---- caching helpers --------------------------------------------------

    @staticmethod
    def save_to_parquet(vectors: np.ndarray, path: Path):
        df = pd.DataFrame(vectors)
        df.columns = [f"f{i}" for i in df.columns]
        df.to_parquet(path)

    @staticmethod
    def load_from_parquet(path: Path) -> np.ndarray:
        return pd.read_parquet(path).to_numpy(dtype=np.float32)


def embed_or_load(texts: list[str], cache_name: str = "bert.parquet",
                  embedder: BertEmbedder | None = None) -> np.ndarray:
    """One-stop helper - returns cached embeddings if present, otherwise
    runs BERT, saves the result, and returns the array."""
    cache = config.CACHE_DIR / cache_name
    if cache.exists():
        return BertEmbedder.load_from_parquet(cache)
    embedder = embedder or BertEmbedder()
    vecs = embedder.encode(texts)
    BertEmbedder.save_to_parquet(vecs, cache)
    return vecs
