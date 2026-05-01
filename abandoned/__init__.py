"""
Modules in this sub-package were tried during development but didn't make
it into the final F-EBA pipeline. They are kept so the dissertation's
discussion of what was tried and why it was dropped is reproducible.
"""

from .flickr_captioner import Flickr8kCaptioner
from .glove            import GloveEmbedder

__all__ = ["Flickr8kCaptioner", "GloveEmbedder"]
