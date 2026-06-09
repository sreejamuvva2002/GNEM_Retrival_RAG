"""Dynamic KB metadata providers for value resolution."""
from georgia_ev_intelligence.shared import config

from .file_provider import FileMetadataProvider
from .live_provider import LiveMetadataProvider, build_live_index
from .provider import (
    DEFAULT_FIELD_ALIASES,
    ColumnMetaView,
    IndexBackedProvider,
    MetadataProvider,
)


def build_metadata_provider() -> MetadataProvider:
    """Construct the configured provider (``METADATA_PROVIDER``: 'live' | 'file')."""
    if str(config.METADATA_PROVIDER).lower() == "file":
        return FileMetadataProvider.from_path(config.METADATA_SNAPSHOT_PATH)
    return LiveMetadataProvider()


__all__ = [
    "MetadataProvider",
    "ColumnMetaView",
    "IndexBackedProvider",
    "DEFAULT_FIELD_ALIASES",
    "LiveMetadataProvider",
    "FileMetadataProvider",
    "build_live_index",
    "build_metadata_provider",
]
