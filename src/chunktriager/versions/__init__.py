"""Game-version profiles.

Every version-dependent detail (folder layout, chunk NBT layout, accepted
DataVersion range) lives in a profile. Code elsewhere asks the profile and
never branches on version numbers itself. Add a profile module and register
it here to support another version.
"""

from __future__ import annotations

from .base import BlockEntity, ChunkInfo, UnsupportedChunkError, VersionProfile
from .java_26_2 import PROFILE as _JAVA_26_2

_PROFILES: dict[str, VersionProfile] = {p.name: p for p in (_JAVA_26_2,)}


class UnknownVersionError(Exception):
    pass


def get_profile(name: str) -> VersionProfile:
    try:
        return _PROFILES[name]
    except KeyError:
        supported = ", ".join(sorted(_PROFILES))
        raise UnknownVersionError(
            f"unsupported game_version {name!r}; supported: {supported}"
        ) from None


def supported_versions() -> list[str]:
    return sorted(_PROFILES)


__all__ = [
    "BlockEntity",
    "ChunkInfo",
    "UnknownVersionError",
    "UnsupportedChunkError",
    "VersionProfile",
    "get_profile",
    "supported_versions",
]
