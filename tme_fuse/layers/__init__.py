"""
TME-FUSE Geometric & Cross-Modal Neural Layers.
Includes Norm-Clamped E(3)-Equivariant Graph Neural Network (EGNN) and BioCLIP Cross-Attention Bridge.
"""

from .egnn_clamped import NormClampedEGNNLayer
from .cross_attention import CrossAttentionLayer
from .text_encoder import BioCLIPTextEncoder

__all__ = ["NormClampedEGNNLayer", "CrossAttentionLayer", "BioCLIPTextEncoder"]
