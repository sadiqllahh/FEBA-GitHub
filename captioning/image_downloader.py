"""
Pull images from Twitter media URLs.

Each image is named after its tweet id so the row -> file mapping is kept.
We split downloads into per-class folders (depressed / non_depressed) so the
folder structure itself encodes the label.
"""

from __future__ import annotations

import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import config


CLASS_DIRS = {0: "non_depressed", 1: "depressed"}


class ImageDownloader:
    """
    Downloads tweet images concurrently.

    Tweets that no longer exist (deleted media) are skipped silently - the
    network failure rate on Twitter URLs is high enough that exception spam
    is more noise than signal.
    """

    def __init__(self,
                 root: Path | None = None,
                 max_workers: int = 16,
                 timeout: int = 10):
        self.root = Path(root or config.IMAGES_DIR)
        self.max_workers = max_workers
        self.timeout = timeout
        for sub in CLASS_DIRS.values():
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _target_path(self, tweet_id: str | int, label: int) -> Path:
        return self.root / CLASS_DIRS[label] / f"{tweet_id}.jpg"

    def _download_one(self, tweet_id, url, label) -> Path | None:
        target = self._target_path(tweet_id, label)
        if target.exists():
            return target
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                target.write_bytes(r.read())
            return target
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None

    def download_all(self, df, id_col="TweetId", url_col="image", label_col="Label"):
        """
        Returns a list of (tweet_id, local_path|None) tuples in the same order
        as the input dataframe so callers can join it back.
        """
        results: list[tuple] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._download_one, row[id_col], row[url_col], int(row[label_col])): row[id_col]
                for _, row in df.iterrows()
                if isinstance(row[url_col], str) and row[url_col].startswith("http")
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                results.append((tid, fut.result()))
        return results
