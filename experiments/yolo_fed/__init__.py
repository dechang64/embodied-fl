"""
Embodied-FL: Federated Object Detection
=========================================
Paper-ready experiment framework.

Architecture:
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  Factory A   │     │                  │     │  Factory C   │
  │  (local head)│────►│  Global Server   │────►│  (local head)│
  │  Backbone ▲  │     │  Backbone Agg    │     │  Backbone ▲  │
  └──────────────┘     └──────────────────┘     └──────────────┘
       ▲                      ▲                       ▲
       └──────────────────────┴───────────────────────┘
              Shared backbone, local detection heads

Key contributions:
  1. Backbone-only aggregation for heterogeneous detection tasks
  2. Task-Aware weighting (cosine similarity + performance + sample size)
  3. Per-client head persistence across rounds

Usage:
  python experiment.py --mode quick    # ~60s on CPU (proof of concept)
  python experiment.py --mode paper    # ~30min on GPU (paper results)
  python experiment.py --mode full     # ~2hr on GPU (all ablations)
"""

__version__ = "1.0.0"
__author__ = "Embodied-FL Team"
