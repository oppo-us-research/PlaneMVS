"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

from .planercnn import PlaneRCNN
from .planestereo import PlaneStereo
from .planercnn_refine import PlaneRCNNRefine
from .planercnn_consistency import PlaneRCNNConsistency
from .planestereo_finetune import PlaneStereo_Finetune
from .planedepthstereo import PlaneDepthStereo


_DETECTION_META_ARCHITECTURES = {
            "PlaneRCNN": PlaneRCNN,
            "PlaneRCNNRefine": PlaneRCNNRefine,
            "PlaneRCNNConsistency": PlaneRCNNConsistency,
            "PlaneStereo": PlaneStereo, # ours;
            "PlaneStereo_Finetune": PlaneStereo_Finetune,
            "PlaneDepthStereo": PlaneDepthStereo
        }


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)

