"""
Baseline Models for Comparison with CoFARS-Sparse
"""

from .average_pooling import AveragePoolingBaseline
from .standard_gru import StandardGRUBaseline
from .din import DINBaseline

__all__ = [
    'AveragePoolingBaseline',
    'StandardGRUBaseline', 
    'DINBaseline'
]
