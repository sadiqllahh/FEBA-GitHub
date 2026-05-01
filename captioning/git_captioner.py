"""
Caption tweet images with the GIT (Generative Image-to-text Transformer)
model. We use the COCO-tuned checkpoint from HuggingFace - same family of
weights described in the report, just exposed via the `transformers` API.

Captions go through the regular text preprocessing pipeline afterwards
(lemma + stem + stop-word removal) before being concatenated with the tweet
text for embedding.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .. import config
from preprocessing import TextCleaner, TextTransformer


class GITCaptioner:
    """
    Wrapper around HuggingFace's GIT.

    Holds a single processor + model on the chosen device and exposes a
    `caption_path()` method for one image and `caption_batch()` for a list.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = model_name or config.GIT_MODEL_NAME
        self.processor = AutoProcessor.from_pretrained(name)
        self.model     = AutoModelForCausalLM.from_pretrained(name).to(self.device)
        self.model.eval()

        # Captions are dirty too - same cleaner the tweets get
        self._cleaner = TextCleaner()
        self._tx      = TextTransformer()

    def _load_image(self, path: str | Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img.resize(config.IMAGE_RESIZE)

    @torch.inference_mode()
    def caption_batch(self, paths: list[str | Path], max_length: int = 32) -> list[str]:
        images = [self._load_image(p) for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        ids    = self.model.generate(pixel_values=inputs.pixel_values,
                                     max_length=max_length)
        raw    = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [self._postprocess(c) for c in raw]

    def caption_path(self, path: str | Path) -> str:
        return self.caption_batch([path])[0]

    # ----------------------------------------------------------------------

    def _postprocess(self, caption: str) -> str:
        """Run the caption through the same text pipeline as tweets so
        captions and tweets share a vocabulary."""
        return self._tx.transform(self._cleaner.clean(caption))


def caption_dataframe(df, captioner: GITCaptioner | None = None,
                      path_col: str = "image_path",
                      out_col: str  = "caption",
                      batch_size: int = 16):
    """
    Add a caption column to `df`. Rows without an image (NaN path) get an
    empty string so downstream concatenation still works.
    """
    captioner = captioner or GITCaptioner()
    captions: list[str] = []
    paths = df[path_col].tolist()

    for i in range(0, len(paths), batch_size):
        chunk = [p for p in paths[i:i + batch_size] if isinstance(p, (str, Path)) and Path(p).exists()]
        if not chunk:
            captions.extend([""] * len(paths[i:i + batch_size]))
            continue
        produced = captioner.caption_batch(chunk)
        # re-align with original positions (some may have been missing)
        idx = 0
        for p in paths[i:i + batch_size]:
            if isinstance(p, (str, Path)) and Path(p).exists():
                captions.append(produced[idx])
                idx += 1
            else:
                captions.append("")
    df[out_col] = captions
    return df
