"""NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.

A minimal, teaching-oriented PyTorch re-implementation of Mildenhall et al. (ECCV 2020).
Supports the Blender synthetic dataset (nerf_synthetic) and covers all core mechanisms:
positional encoding, coarse/fine MLPs, stratified & hierarchical sampling, differentiable
volume rendering, and the coarse+fine training objective.
"""

__version__ = "0.1.0"
