"""
Run the same evaluation pipeline on the Kyoto generalisation dataset.

The path lives in `config.KYOTO_PATH`. The Kyoto CSV is expected to share
the schema of the training datasets (TweetId, Tweet, Label, image), so we
can reuse the preprocessing + feature pipelines verbatim. The label set is
{depressed, non_depressed, candidate}; we collapse 'candidate' down to
non_depressed for the binary evaluation, matching the way the original
F-EBA was trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from preprocessing import clean_dataframe, transform_dataframe
from features import (
    Word2VecEmbedder, tokenise,
    SelfAttention, BertEmbedder,
)
from .evaluation    import evaluate, plot_confusion, plot_roc, plot_prc
from models import FEBA
from adversarial import AdversarialTrainer
from models.bilstm import BiLSTMClassifier


# ---------------------------------------------------------------------------

LABEL_MAP = {
    "depressed":      1,
    "non_depressed":  0,
    "candidate":      0,                      # collapsed to non-depressed
    1: 1, 0: 0,
}


@dataclass
class GeneralisationResults:
    """Container for the four model variants we benchmark on Kyoto."""
    feba_optimised:   "EvalReport"   # F-EBA on optimised feature set
    feba_bow:         "EvalReport"   # F-EBA on Bag-of-Words baseline
    feba_no_defense:  "EvalReport"   # F-EBA without adversarial layer
    feba_with_defense:"EvalReport"   # F-EBA + FGSM defense


# ---------------------------------------------------------------------------

def load_kyoto(path: str | Path | None = None) -> pd.DataFrame:
    """Reads the Kyoto CSV and normalises its columns / labels."""
    df = pd.read_csv(path or config.KYOTO_PATH)
    expected = {"TweetId", "Tweet", "Label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Kyoto CSV missing columns: {missing}")
    df["Label"] = df["Label"].map(LABEL_MAP).astype(int)
    return df


# ---------------------------------------------------------------------------

def _bow_features(corpus: list[str], vocab_size: int = 5000) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(max_features=vocab_size)
    return vec.fit_transform(corpus).toarray().astype(np.float32)


def build_optimised_features(df: pd.DataFrame,
                             w2v: Word2VecEmbedder,
                             bert: BertEmbedder) -> np.ndarray:
    """
    Recreates the feature pipeline used at training time: Word2Vec average
    + self-attention + BERT, all concatenated into one vector per tweet.
    """
    tokens = tokenise(df["Tweet"])
    w2v_vecs = w2v.embed_corpus(tokens)

    # self-attention pass over each tweet's averaged vector
    sa = SelfAttention(input_dim=1, hidden_dim=8)
    attended = np.array([
        sa(v.reshape(-1, 1)).mean(axis=0) for v in w2v_vecs
    ])

    bert_vecs = bert.encode(df["Tweet"].tolist())
    return np.hstack([w2v_vecs, attended, bert_vecs])


# ---------------------------------------------------------------------------

def run_generalisation(kyoto_path: str | Path,
                       trained_w2v: Word2VecEmbedder,
                       trained_bert: BertEmbedder,
                       trained_feba: FEBA,
                       feature_indices: np.ndarray | None = None
                       ) -> GeneralisationResults:
    """
    The big one. Loads Kyoto, runs all four reported experiments, returns
    the report bundle plus saves the plots into config.CACHE_DIR / "kyoto".
    """
    out_dir = config.CACHE_DIR / "kyoto"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Kyoto dataset...")
    df = load_kyoto(kyoto_path)
    df = clean_dataframe(df)
    df = transform_dataframe(df)

    y = df["Label"].to_numpy()

    # ---- 1. F-EBA on optimised features ---------------------------------
    print("Building optimised features...")
    X_opt = build_optimised_features(df, trained_w2v, trained_bert)
    if feature_indices is not None:
        X_opt = X_opt[:, feature_indices]

    print("F-EBA (optimised) inference...")
    proba = trained_feba.predict_proba(X_opt)[:, 1]
    preds = trained_feba.predict(X_opt)
    feba_optimised = evaluate(y, preds, y_proba=proba)
    _save_plots(feba_optimised, out_dir, "feba_optimised")

    # ---- 2. F-EBA on BOW baseline ---------------------------------------
    print("F-EBA (BOW) re-train + eval...")
    X_bow = _bow_features(df["Tweet"].tolist())
    bow_model = FEBA(input_dim=X_bow.shape[1], n_models=config.FEBA_N_MODELS)
    val_split = int(0.85 * len(X_bow))
    bow_model.train_weak_classifiers(
        X_bow[:val_split], y[:val_split],
        X_bow[val_split:], y[val_split:],
    )
    feba_bow = evaluate(
        y, bow_model.predict(X_bow),
        y_proba=bow_model.predict_proba(X_bow)[:, 1],
    )
    _save_plots(feba_bow, out_dir, "feba_bow")

    # ---- 3. Adversarial layer comparison --------------------------------
    print("Adversarial defense comparison...")
    feba_no_defense   = feba_optimised   # already computed - same model

    bilstm_defended = BiLSTMClassifier(input_dim=X_opt.shape[1])
    trainer = AdversarialTrainer(bilstm_defended)
    trainer.fit(X_opt[:val_split], y[:val_split], epochs=config.FEBA_EPOCHS)

    import torch
    bilstm_defended.eval()
    with torch.no_grad():
        logits = bilstm_defended(torch.tensor(X_opt, dtype=torch.float32))
        defended_probs = logits.softmax(-1).numpy()
    feba_with_defense = evaluate(
        y, defended_probs.argmax(1), y_proba=defended_probs[:, 1],
    )
    _save_plots(feba_with_defense, out_dir, "feba_with_defense")

    return GeneralisationResults(
        feba_optimised   = feba_optimised,
        feba_bow         = feba_bow,
        feba_no_defense  = feba_no_defense,
        feba_with_defense= feba_with_defense,
    )


# ---------------------------------------------------------------------------

def _save_plots(report, out_dir: Path, prefix: str):
    """Three plots per model variant: confusion matrix, ROC, PRC."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    plot_confusion(report, ax=ax[0], title=f"{prefix} - confusion")
    plot_roc(report, label=prefix, ax=ax[1])
    plot_prc(report, label=prefix, ax=ax[2])
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}.png", dpi=150)
    plt.close(fig)
