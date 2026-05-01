"""
Step (b) of preprocessing - clean raw tweet text.

Order matters here. We pull emojis out first (they encode emotion), then
strip the things that aren't worth keeping (mentions, links), then handle
special characters while preserving curse words because curse-word frequency
is itself a useful signal for depression detection.
"""

import re
import demoji

from .. import config


# Pre-compiled patterns. Compiling once is faster across millions of tweets.
_MENTION_RE = re.compile(r"@\w+")
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_RT_RE      = re.compile(r"\bRT\b[: ]*")          # leading RT artifact
_HASHTAG_RE = re.compile(r"#(\w+)")               # keep the word, drop the #
_NON_ALNUM  = re.compile(r"[^a-zA-Z\s]")          # punctuation, symbols, digits
_MULTI_WS   = re.compile(r"\s+")


class TextCleaner:
    """
    Cleans a single tweet.

    The cleaner deliberately keeps:
      - Emoji meanings (converted to words via demoji)
      - Curse words (potential depression signal - see report)
      - Hashtag content (e.g. #depressed -> "depressed")

    It strips: user mentions, URLs, retweet markers, punctuation, digits,
    and collapses whitespace.
    """

    def __init__(self, curse_words: set[str] | None = None):
        self.curse_words = curse_words or config.CURSE_WORDS
        # demoji caches its codepoint table on first call
        demoji.download_codes() if not demoji._EMOJI_PAT else None  # noqa

    # ---- individual passes ------------------------------------------------

    def emojis_to_text(self, text: str) -> str:
        """Replace every emoji with its CLDR short name in spaces."""
        return demoji.replace_with_desc(text, sep=" ")

    def strip_handles_and_urls(self, text: str) -> str:
        text = _MENTION_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        text = _RT_RE.sub(" ", text)
        return text

    def strip_special_chars(self, text: str) -> str:
        """
        Remove punctuation, digits and non-alphabetic noise but keep curse
        words intact. Curse words are protected by tagging them before the
        scrub and untagging after.
        """
        # protect curse words with sentinel markers that survive scrubbing
        protected = text
        for w in self.curse_words:
            protected = re.sub(rf"\b{re.escape(w)}\b",
                               f"__CW_{w}__", protected, flags=re.IGNORECASE)

        # un-hash hashtags before scrubbing so the word survives
        protected = _HASHTAG_RE.sub(r"\1", protected)
        protected = _NON_ALNUM.sub(" ", protected)

        # restore curse words
        for w in self.curse_words:
            protected = protected.replace(f"__CW_{w}__", w)

        return protected

    # ---- driver -----------------------------------------------------------

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self.emojis_to_text(text)
        text = self.strip_handles_and_urls(text)
        text = self.strip_special_chars(text)
        text = _MULTI_WS.sub(" ", text).strip().lower()
        return text


def clean_dataframe(df, text_col: str = "Tweet", lang_col: str | None = None):
    """
    Apply TextCleaner to a whole dataframe and (optionally) drop non-English
    rows. The Shen et al. and Clpsych dumps both contain non-English tweets
    that we don't want.
    """
    cleaner = TextCleaner()

    if lang_col and lang_col in df.columns:
        df = df[df[lang_col] == config.LANGUAGE_FILTER].copy()

    df[text_col] = df[text_col].astype(str).map(cleaner.clean)
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    return df
