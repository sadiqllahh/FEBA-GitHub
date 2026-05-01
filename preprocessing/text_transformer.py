"""
Step (c) of preprocessing - transform cleaned text.

Lemmatize, stem, drop stop words, mark negations, and keep curse words as-is.
Stemming is mildly destructive (the 'the' -> 'thi' issue mentioned in the
report) so we lemmatize first, then stem.
"""

from __future__ import annotations

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from .. import config
from .contractions import expand_contractions


# NLTK has lazy resources - make sure they're present
for _pkg in ("punkt", "wordnet", "stopwords", "omw-1.4"):
    try:
        nltk.data.find(_pkg)
    except LookupError:
        nltk.download(_pkg, quiet=True)


# stop words minus negations (we want to keep negations for sentiment)
_STOPWORDS = set(stopwords.words("english")) - config.NEGATION_TOKENS

# A handful of mis-stems we actively want to undo. The report mentions
# 'the' -> 'thi'; add others as you spot them.
_STEM_FIXES = {
    "thi": "the",
    "wa":  "was",
    "ha":  "has",
    "doe": "does",
}


class TextTransformer:
    """
    Lemmatize -> stem -> filter stop words, with curse words protected and
    negations tagged onto the next token (e.g. 'not happy' -> 'not_happy')
    so they survive downstream tokenisation.
    """

    def __init__(self,
                 keep_curse_words: bool = True,
                 mark_negations: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer    = PorterStemmer()
        self.keep_curse_words = keep_curse_words
        self.mark_negations   = mark_negations
        self.curse_words = config.CURSE_WORDS

    # ----------------------------------------------------------------------

    def _stem(self, token: str) -> str:
        stemmed = self.stemmer.stem(token)
        return _STEM_FIXES.get(stemmed, stemmed)

    def _process_token(self, token: str) -> str:
        if token in self.curse_words and self.keep_curse_words:
            return token                                    # leave intact
        token = self.lemmatizer.lemmatize(token)
        token = self._stem(token)
        return token

    def _attach_negations(self, tokens: list[str]) -> list[str]:
        """
        Walk the token list once. When we see a negation we glue it onto the
        next non-stopword token. This stops 'not good' from collapsing into
        an unrelated 'good'.
        """
        out, i = [], 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in config.NEGATION_TOKENS and i + 1 < len(tokens):
                out.append(f"{tok}_{tokens[i + 1]}")
                i += 2
            else:
                out.append(tok)
                i += 1
        return out

    # ----------------------------------------------------------------------

    def transform(self, text: str) -> str:
        text   = expand_contractions(text.lower())
        tokens = nltk.word_tokenize(text)

        # drop stop words first (cheaper than lemmatising things we'll throw
        # away). Negations are kept because we removed them from _STOPWORDS.
        tokens = [t for t in tokens if t not in _STOPWORDS and t.isalpha()]

        if self.mark_negations:
            tokens = self._attach_negations(tokens)

        tokens = [self._process_token(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 1]
        return " ".join(tokens)


def transform_dataframe(df, text_col: str = "Tweet"):
    """In-place column rewrite for a whole dataframe."""
    tx = TextTransformer()
    df[text_col] = df[text_col].astype(str).map(tx.transform)
    return df[df[text_col].str.len() > 0].reset_index(drop=True)
