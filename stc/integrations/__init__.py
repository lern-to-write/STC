"""Framework adapters that bind STC primitives to specific model libraries.

The only adapter currently shipped is :mod:`stc.integrations.hf_vit`, which
patches HuggingFace pre-LN ViT vision towers (CLIP & SigLIP).  See its
docstring for the contract a vision tower must satisfy.
"""

from stc.integrations.hf_vit import (
    register_stc_cacher,
    unregister_stc_cacher,
)

__all__ = ["register_stc_cacher", "unregister_stc_cacher"]
