"""
Word2Vec model training + embedding lookup.

Training is over the union of tweet text and image captions from both
datasets, exactly as described in the report. Once trained, the model is
pickled to disk so subsequent runs skip retraining.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from gensim.models import Word2Vec

from .. import config


class Word2VecEmbedder:
    """
    Trains (or loads) a Word2Vec model and produces a fixed-size vector for
    each tweet by averaging its word vectors. Words missing from the
    vocabulary contribute a zero vector, which is the standard fallback for
    OOV tokens.
    """

    DEFAULT_CACHE = config.CACHE_DIR / "word2vec.model"

    def __init__(self, dim: int | None = None, cache: Path | None = None):
        self.dim   = dim or config.WORD2VEC_DIM
        self.cache = Path(cache or self.DEFAULT_CACHE)
        self.model: Word2Vec | None = None

    # ---- training & loading ----------------------------------------------

    def train(self, sentences: list[list[str]]) -> "Word2VecEmbedder":
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.dim,
            window=config.WORD2VEC_WINDOW,
            min_count=config.WORD2VEC_MIN_COUNT,
            epochs=config.WORD2VEC_EPOCHS,
            sg=1,                                # skip-gram, slightly better on tweets
            workers=4,
            seed=config.RANDOM_SEED,
        )
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.cache))
        return self

    def load(self) -> "Word2VecEmbedder":
        self.model = Word2Vec.load(str(self.cache))
        return self

    def fit_or_load(self, sentences: list[list[str]]) -> "Word2VecEmbedder":
        return self.load() if self.cache.exists() else self.train(sentences)

    # ---- embedding --------------------------------------------------------

    def embed(self, tokens: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        kv = self.model.wv
        vecs = [kv[t] for t in tokens if t in kv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def embed_corpus(self, sentences: list[list[str]]) -> np.ndarray:
        return np.vstack([self.embed(s) for s in sentences])


def tokenise(series) -> list[list[str]]:
    """Cheap whitespace tokeniser. Tweets are already cleaned at this point."""
    return [str(s).split() for s in series]
