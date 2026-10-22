"""
TME-FUSE Geometric & Cross-Modal Neural Layers.
Includes Norm-Clamped E(3)-Equivariant Graph Neural Network (EGNN) and Clinical Text Conditioning Bridge.
"""

from .egnn_clamped import NormClampedEGNNLayer
from .cross_attention import CrossAttentionLayer, AdditivePromptProjectionLayer
from .text_encoder import ClinicalTextEncoder, BioCLIPTextEncoder

__all__ = ["NormClampedEGNNLayer", "CrossAttentionLayer", "AdditivePromptProjectionLayer", "ClinicalTextEncoder", "BioCLIPTextEncoder"]
