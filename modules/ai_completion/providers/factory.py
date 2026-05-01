"""Resolve the provider name (env-driven) into a concrete instance."""
from __future__ import annotations

import logging
from typing import Optional

from ..base import CompletionProvider

logger = logging.getLogger(__name__)


def build_provider(provider_name: Optional[str] = None) -> CompletionProvider:
    """
    Map a string id to a provider instance.  Unknown names → NoneProvider.
    Names accepted (case-insensitive):
        none | hunyuan3d_replicate | hunyuan3d | meshy
    """
    name = (provider_name or "none").strip().lower()

    if name in ("none", "", "off", "disabled"):
        from .none_provider import NoneProvider
        return NoneProvider()

    if name in ("hunyuan3d", "hunyuan3d_replicate", "hunyuan", "tencent_hunyuan3d"):
        from .hunyuan3d_replicate import Hunyuan3DReplicateProvider
        return Hunyuan3DReplicateProvider()

    if name in ("meshy", "meshy_ai"):
        from .meshy_provider import MeshyProvider
        return MeshyProvider()

    logger.warning(f"Unknown ai-completion provider '{name}', falling back to NoneProvider")
    from .none_provider import NoneProvider
    return NoneProvider()
