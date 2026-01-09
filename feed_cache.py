#!/usr/bin/env python3
"""
Feed caching with HTTP conditional request support.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CacheEntry:
    """Represents a cached feed response."""
    url: str
    content: bytes
    etag: Optional[str]
    last_modified: Optional[str]
    cached_at: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cache entry has expired based on TTL."""
        return time.time() > (self.cached_at + self.ttl_seconds)


class FeedCache:
    """File-based cache for RSS feed responses."""

    def __init__(self, cache_dir: str = ".rss_cache", ttl_minutes: int = 15):
        """
        Initialize feed cache.

        Args:
            cache_dir: Directory to store cached feeds
            ttl_minutes: Time-to-live for cached entries in minutes
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_minutes * 60
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, url: str) -> str:
        """Generate cache file path from URL hash."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{url_hash}.cache")

    def _get_meta_path(self, url: str) -> str:
        """Generate metadata file path."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{url_hash}.meta.json")

    def get(self, url: str) -> Optional[CacheEntry]:
        """
        Retrieve cached entry if it exists.

        Args:
            url: Feed URL

        Returns:
            CacheEntry if found, None otherwise
        """
        meta_path = self._get_meta_path(url)
        cache_path = self._get_cache_path(url)

        if not os.path.exists(meta_path) or not os.path.exists(cache_path):
            return None

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            with open(cache_path, 'rb') as f:
                content = f.read()

            entry = CacheEntry(
                url=meta['url'],
                content=content,
                etag=meta.get('etag'),
                last_modified=meta.get('last_modified'),
                cached_at=meta['cached_at'],
                ttl_seconds=meta['ttl_seconds']
            )
            return entry
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def set(self, url: str, content: bytes,
            etag: Optional[str] = None,
            last_modified: Optional[str] = None) -> None:
        """
        Store feed response in cache.

        Args:
            url: Feed URL
            content: Feed content (raw bytes)
            etag: Optional ETag header from response
            last_modified: Optional Last-Modified header from response
        """
        cache_path = self._get_cache_path(url)
        meta_path = self._get_meta_path(url)

        meta = {
            'url': url,
            'etag': etag,
            'last_modified': last_modified,
            'cached_at': time.time(),
            'ttl_seconds': self.ttl_seconds
        }

        with open(cache_path, 'wb') as f:
            f.write(content)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def get_conditional_headers(self, url: str) -> dict:
        """
        Get headers for conditional HTTP request.

        Args:
            url: Feed URL

        Returns:
            Dictionary with If-None-Match and/or If-Modified-Since headers
        """
        entry = self.get(url)
        headers = {}
        if entry:
            if entry.etag:
                headers['If-None-Match'] = entry.etag
            if entry.last_modified:
                headers['If-Modified-Since'] = entry.last_modified
        return headers

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of files removed
        """
        count = 0
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(filepath)
                    count += 1
                except OSError:
                    pass
        return count
