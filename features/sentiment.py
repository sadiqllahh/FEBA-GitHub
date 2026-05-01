"""
Sentiment scoring via the StanfordCoreNLP Java server.

Run the server beforehand, e.g.:

    java -mx4g -cp 'stanford-corenlp-4.5.5/*' \
        edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
        -port 9000 -timeout 15000

The wrapper returns a score in {0, 1, 2, 3, 4} where 0 is most-negative,
2 neutral, 4 most-positive (matches the report).
"""

from __future__ import annotations

from stanfordcorenlp import StanfordCoreNLP
import json

from .. import config


class SentimentScorer:
    """
    Thin wrapper around StanfordCoreNLP.

    The connection is lazy - you can construct the object freely and we only
    actually open the socket when a tweet is scored. Use it as a context
    manager (`with SentimentScorer() as s: ...`) so the underlying CoreNLP
    process is shut down cleanly at the end.
    """

    _PROPS = {
        "annotators": "tokenize,ssplit,parse,sentiment",
        "outputFormat": "json",
        "timeout": "15000",
    }

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or config.SENTIMENT_SERVER
        self.port = port or config.SENTIMENT_PORT
        self._nlp: StanfordCoreNLP | None = None

    # ---- connection management --------------------------------------------

    def _ensure(self):
        if self._nlp is None:
            self._nlp = StanfordCoreNLP(self.host, port=self.port)

    def close(self):
        if self._nlp is not None:
            self._nlp.close()
            self._nlp = None

    def __enter__(self):
        self._ensure()
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- scoring ----------------------------------------------------------

    def score(self, text: str) -> int:
        """
        Returns 0..4. If CoreNLP can't parse the input (empty, weird tokens),
        we default to 2 (neutral) rather than raising - it lets us batch
        across millions of tweets without a single bad row killing the run.
        """
        if not text.strip():
            return 2
        self._ensure()
        try:
            raw = self._nlp.annotate(text, properties=self._PROPS)
            doc = json.loads(raw)
            sentences = doc.get("sentences", [])
            if not sentences:
                return 2
            # CoreNLP gives one sentiment per sentence. Tweets are short, so
            # average over them and round to the nearest integer score.
            scores = [int(s["sentimentValue"]) for s in sentences]
            return round(sum(scores) / len(scores))
        except Exception:
            return 2

    def score_many(self, texts: list[str]) -> list[int]:
        return [self.score(t) for t in texts]


def add_sentiment_column(df, text_col: str = "Tweet", out_col: str = "sentiment"):
    """Adds a 0..4 sentiment score column. Pure convenience wrapper."""
    with SentimentScorer() as scorer:
        df[out_col] = [scorer.score(t) for t in df[text_col].astype(str)]
    return df
