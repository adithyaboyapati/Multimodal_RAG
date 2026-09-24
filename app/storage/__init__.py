"""
Asset storage and artifact persistence package.
"""
from app.storage.asset_store import AssetStore, get_asset_store

__all__ = ["AssetStore", "get_asset_store"]
