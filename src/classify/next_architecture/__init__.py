"""Leakage-resistant next-architecture pipeline (E0–E5, E8).

Does not write canonical Branch A/B/C artifacts.
"""

from src.classify.next_architecture.mil import mil_readiness

__all__ = ["mil_readiness"]
