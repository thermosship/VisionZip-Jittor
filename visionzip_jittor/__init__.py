"""Native Jittor reproduction utilities for VisionZip.

The Jittor dependency is imported lazily so the PyTorch reference tools can be
used in a separate environment without installing Jittor there.
"""

from .config import VisionZipConfig, load_config
from .projector_config import ProjectorConfig, load_projector_config

__all__ = [
    "VisionZipConfig",
    "load_config",
    "ProjectorConfig",
    "load_projector_config",
]
__version__ = "0.1.0"
