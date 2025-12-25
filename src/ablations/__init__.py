"""
Ablation Study Variants
Testing contribution of each component
"""

from .no_prototypes import CoFARSSparseNoPrototypes
from .no_hybrid import CoFARSSparseNoHybrid
from .no_independence import CoFARSSparseNoIndependence

__all__ = [
    'CoFARSSparseNoPrototypes',
    'CoFARSSparseNoHybrid',
    'CoFARSSparseNoIndependence'
]
