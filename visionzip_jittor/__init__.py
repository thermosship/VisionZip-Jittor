"""Native Jittor reproduction utilities for VisionZip.

The Jittor dependency is imported lazily so the PyTorch reference tools can be
used in a separate environment without installing Jittor there.
"""

from .config import VisionZipConfig, load_config
from .gpt2_config import (
    GPT2Config,
    Phase3BConfig,
    load_gpt2_config,
    load_phase3b_config,
)
from .phase4_config import Phase4AConfig, load_phase4a_config
from .phase4_data import PairedManifest, PairedSample, load_paired_manifest
from .phase4b_config import Phase4BConfig, load_phase4b_config
from .phase4b_data import (
    Phase4BPreparedManifest,
    Phase4BPreparedSample,
    load_prepared_dataset,
)
from .phase4b_features import Phase4BFeatureManifest, load_feature_manifest
from .projector_config import ProjectorConfig, load_projector_config

__all__ = [
    "VisionZipConfig",
    "load_config",
    "ProjectorConfig",
    "load_projector_config",
    "GPT2Config",
    "Phase3BConfig",
    "load_gpt2_config",
    "load_phase3b_config",
    "Phase4AConfig",
    "load_phase4a_config",
    "PairedManifest",
    "PairedSample",
    "load_paired_manifest",
    "Phase4BConfig",
    "load_phase4b_config",
    "Phase4BPreparedManifest",
    "Phase4BPreparedSample",
    "load_prepared_dataset",
    "Phase4BFeatureManifest",
    "load_feature_manifest",
]
__version__ = "0.1.0"
