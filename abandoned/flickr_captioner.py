"""
Initial image-captioning attempt (Flickr-8k) - abandoned.

The report explains: "due to the limited number of images in the Flickr 80
dataset, the captioning model's performance could have been better." We
keep the implementation here for completeness; it is not used by the main
pipeline (see captioning.git_captioner for the replacement).

The encoder is a frozen ResNet-50; the decoder is a simple LSTM over the
caption vocabulary. Trained for ~20 epochs on Flickr-8k captions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image


class Flickr8kCaptioner(nn.Module):
    """
    Encoder-decoder captioner. Kept around so we can rerun the original
    experiment if anyone asks.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 1):
        super().__init__()

        encoder = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.fc = nn.Linear(encoder.fc.in_features, embed_dim)
        self.encoder = encoder

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.head      = nn.Linear(hidden_dim, vocab_size)

        self._tx = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ---- training side ----------------------------------------------------

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        feats   = self.encoder(images).unsqueeze(1)            # (B, 1, E)
        embeds  = self.embedding(captions[:, :-1])             # teacher forcing
        seq     = torch.cat([feats, embeds], dim=1)
        out, _  = self.lstm(seq)
        return self.head(out)

    # ---- inference --------------------------------------------------------

    @torch.inference_mode()
    def caption(self, image_path: str, idx_to_word: dict[int, str],
                start_token: int, end_token: int, max_len: int = 20) -> str:
        img = self._tx(Image.open(image_path).convert("RGB")).unsqueeze(0)
        feats = self.encoder(img).unsqueeze(1)
        word  = torch.tensor([[start_token]])
        h     = None
        words: list[str] = []

        for _ in range(max_len):
            embed = self.embedding(word)
            inp   = torch.cat([feats, embed], dim=1) if not words else embed
            out, h = self.lstm(inp, h)
            word  = self.head(out[:, -1]).argmax(-1, keepdim=True)
            tok   = word.item()
            if tok == end_token:
                break
            words.append(idx_to_word.get(tok, ""))
        return " ".join(w for w in words if w)
