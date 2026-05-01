"""
F-EBA: Feature-Enhanced Boosting Algorithm for depression detection on
Twitter. Implementation of the pipeline described in the project report
(data preprocessing -> feature extraction -> RFE -> F-EBA boosting ->
adversarial defense -> SHAP validation -> Kyoto generalisation).
"""

import config
import features
import captioning
import models
import preprocessing
import adversarial, evaluation, xai, generalisation

__all__ = [
    "config",
    "preprocessing", "captioning", "features", "models",
    "adversarial", "evaluation", "xai", "generalisation",
]
