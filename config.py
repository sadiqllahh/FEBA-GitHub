"""
Central place for paths and hyperparameters.

Anything that's likely to change between runs lives here so the rest of the
package doesn't get littered with magic numbers.
"""

from pathlib import Path

# ---- dataset paths ---------------------------------------------------------
# Shen et al. and Clpsych are the two datasets used for training.
# Each CSV is expected to have columns: TweetId, Tweet, Label, image
SHEN_PATH    = Path("data/shen_et_al.csv")
CLPSYCH_PATH = Path("data/clpsych.csv")

# Kyoto generalisation dataset (~1.2M tweets). Path is filled in at run time.
KYOTO_PATH   = Path("data/kyoto.csv")

# image directory used for tweet media downloads
IMAGES_DIR   = Path("data/images")

# where models / embeddings get cached so we don't retrain every run
CACHE_DIR    = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---- preprocessing ---------------------------------------------------------
LANGUAGE_FILTER = "en"           # only keep English tweets

# Curse words are kept on purpose - they're emotional signal, not noise.
# The list below is intentionally short here; extend it for your corpus.
CURSE_WORDS = {
    "fuck", "fucking", "shit", "bitch", "ass", "asshole", "bastard",
    "damn", "crap", "dick", "piss", "hell",
}

NEGATION_TOKENS = {"not", "no", "never", "n't", "without", "neither", "nor"}


# ---- feature extraction ----------------------------------------------------
WORD2VEC_DIM      = 80
WORD2VEC_MIN_COUNT = 2
WORD2VEC_WINDOW    = 5
WORD2VEC_EPOCHS    = 20

GLOVE_DIM        = 80            # used by the abandoned glove module
GLOVE_WINDOW     = 5

BERT_MODEL_NAME  = "bert-base-uncased"
BERT_MAX_LEN     = 64
BERT_HIDDEN_LAYERS = 12          # report mentions 12 in dev, 24 in prod

SENTIMENT_SERVER = "http://localhost"
SENTIMENT_PORT   = 9000

GIT_MODEL_NAME   = "microsoft/git-base-coco"
IMAGE_RESIZE     = (224, 224)


# ---- model / training ------------------------------------------------------
RANDOM_SEED      = 42

FEBA_N_MODELS    = 10            # ten weak classifiers in the ensemble
FEBA_VAL_THRESH  = 0.55          # validation accuracy gate for keeping a learner
FEBA_EPOCHS      = 10
FEBA_BATCH_SIZE  = 64
FEBA_LR          = 1e-3

BILSTM_HIDDEN    = 128
BILSTM_DROPOUT   = 0.3

# RFE feature selection
RFE_N_FEATURES   = 50
TREE_MAX_DEPTH   = 8
TREE_MIN_SPLIT   = 4

# adversarial training
FGSM_EPSILON     = 0.03
ADV_LOSS_WEIGHT  = 0.5           # how much the adversarial loss contributes


# ---- evaluation ------------------------------------------------------------
TRAIN_SPLIT      = 0.7
VAL_SPLIT        = 0.15
# remaining 0.15 is test
CV_FOLDS         = 5
