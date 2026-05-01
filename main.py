"""
End-to-end pipeline runner.

Stages:

  1. Load Shen et al. + Clpsych
  2. Preprocess text (clean -> transform)
  3. Optionally download images and caption them with GIT
  4. Extract features (sentiment, Word2Vec, self-attention, BERT)
  5. Run RFE feature selection
  6. Train F-EBA
  7. Add the adversarial layer
  8. Hyperparameter sweep (normalisation)
  9. SHAP validation
 10. Generalisation on Kyoto

Each stage is its own function so the pipeline is easy to truncate when
debugging - run only the stages you need.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from preprocessing import clean_dataframe, transform_dataframe
from captioning    import ImageDownloader, GITCaptioner, caption_dataframe
from features      import (
    add_sentiment_column,
    Word2VecEmbedder, tokenise,
    SelfAttention,
    BertEmbedder,
    select_features,
)
from models        import FEBA, BoostingClassifier
from adversarial   import AdversarialTrainer
from models.bilstm import BiLSTMClassifier
from evaluation    import evaluate, plot_confusion, plot_roc, plot_prc
from xai           import ShapAnalyzer
from generalisation import run_generalisation
from hyperparameter_tuning import sweep_normalisations


# ---------------------------------------------------------------------------

def load_combined() -> pd.DataFrame:
    """Concatenate Shen et al. + Clpsych into a single dataframe."""
    parts = []
    for p in (config.SHEN_PATH, config.CLPSYCH_PATH):
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError("No training datasets found at the configured paths.")
    return pd.concat(parts, ignore_index=True)


def stage_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_dataframe(df)
    df = transform_dataframe(df)
    df["Label"] = df["Label"].astype(int)
    return df


def stage_caption(df: pd.DataFrame, with_images: bool) -> pd.DataFrame:
    if not with_images or "image" not in df.columns:
        df["caption"] = ""
        return df

    print("Downloading images...")
    downloader = ImageDownloader()
    paths = dict(downloader.download_all(df))
    df["image_path"] = df["TweetId"].map(paths.get)

    print("Generating captions with GIT...")
    captioner = GITCaptioner()
    df = caption_dataframe(df, captioner=captioner, path_col="image_path")
    return df


def stage_features(df: pd.DataFrame) -> tuple[np.ndarray, Word2VecEmbedder, BertEmbedder]:
    # combine tweet text with caption text - the report says the W2V model
    # was trained on both
    corpus = (df["Tweet"].fillna("") + " " + df.get("caption", "").fillna("")).tolist()
    sentences = tokenise(pd.Series(corpus))

    # sentiment
    df = add_sentiment_column(df)

    # word2vec
    w2v = Word2VecEmbedder().fit_or_load(sentences)
    w2v_vecs = w2v.embed_corpus(sentences)

    # self-attention pooling on top of word2vec
    sa = SelfAttention(input_dim=1, hidden_dim=8)
    sa_vecs = np.array([sa(v.reshape(-1, 1)).mean(axis=0) for v in w2v_vecs])

    # BERT
    bert = BertEmbedder()
    bert_vecs = bert.encode(corpus)

    sentiment = df["sentiment"].to_numpy().reshape(-1, 1).astype(np.float32)
    X = np.hstack([sentiment, w2v_vecs, sa_vecs, bert_vecs])
    return X, w2v, bert


def stage_select(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(f"RFE on {X.shape[1]} features -> {config.RFE_N_FEATURES}...")
    return select_features(X, y, n_features=config.RFE_N_FEATURES)


def stage_split(X, y):
    """Stratified-ish split using config splits."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.permutation(len(X))
    n_tr  = int(config.TRAIN_SPLIT * len(X))
    n_val = int(config.VAL_SPLIT   * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def stage_train_feba(X_tr, y_tr, X_val, y_val) -> FEBA:
    feba = FEBA(input_dim=X_tr.shape[1], n_models=config.FEBA_N_MODELS)
    feba.train_weak_classifiers(X_tr, y_tr, X_val, y_val)
    print(f"Kept {len(feba.learners)} / {config.FEBA_N_MODELS} weak learners.")
    return feba


def stage_adversarial(X_tr, y_tr, input_dim) -> BiLSTMClassifier:
    """Train a BiLSTM with the adversarial layer for comparison."""
    model = BiLSTMClassifier(input_dim=input_dim)
    AdversarialTrainer(model).fit(X_tr, y_tr, epochs=config.FEBA_EPOCHS)
    return model


def stage_xai(model, X_train, X_test, feature_names=None):
    sa = ShapAnalyzer(model, background=X_train, feature_names=feature_names)
    return sa.explain(X_test)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="F-EBA pipeline runner")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image download and captioning.")
    parser.add_argument("--kyoto", type=Path,
                        help="Path to the Kyoto generalisation CSV. If set, "
                             "runs the full generalisation suite at the end.")
    parser.add_argument("--norm-sweep", action="store_true",
                        help="Run the normalisation hyperparameter sweep.")
    args = parser.parse_args()

    print("Stage 1: load datasets")
    df = load_combined()

    print("Stage 2: preprocess")
    df = stage_preprocess(df)

    print("Stage 3: image captions")
    df = stage_caption(df, with_images=not args.skip_images)

    print("Stage 4: features")
    X, w2v, bert = stage_features(df)
    y = df["Label"].to_numpy()

    print("Stage 5: feature selection")
    X_red, selected_idx = stage_select(X, y)

    print("Stage 6: split")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = stage_split(X_red, y)

    print("Stage 7: F-EBA")
    feba = stage_train_feba(X_tr, y_tr, X_val, y_val)
    proba = feba.predict_proba(X_te)[:, 1]
    report = evaluate(y_te, feba.predict(X_te), y_proba=proba)
    print(f"F-EBA test accuracy: {report.accuracy:.4f}, "
          f"AUC: {report.roc_auc:.4f}, AP: {report.pr_auc:.4f}")

    print("Stage 8: adversarial defense")
    defended = stage_adversarial(X_tr, y_tr, input_dim=X_tr.shape[1])

    if args.norm_sweep:
        print("Stage 8b: normalisation sweep")
        runs = sweep_normalisations(X_tr, y_tr, X_val, y_val)
        for norm, run in runs.items():
            print(f"  {norm}: final acc = {run.accuracy_per_epoch[-1]:.4f}")

    print("Stage 9: SHAP")
    shap_values = stage_xai(feba, X_tr, X_te[:200])
    print(f"SHAP values shape: {shap_values.shape}")

    if args.kyoto:
        print("Stage 10: Kyoto generalisation")
        run_generalisation(args.kyoto, w2v, bert, feba,
                           feature_indices=selected_idx)


if __name__ == "__main__":
    main()
