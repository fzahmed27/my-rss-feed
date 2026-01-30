#!/usr/bin/env python3
"""
FeedPulse — AI RSS Feed Aggregator Web Interface
Flask web server serving a modern dashboard for AI/ML news.
"""

import datetime
import json
import logging
import os
import shutil
import threading
import time
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request

from ai_rss_aggregator import AINewsAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_aggregator = None
_articles: List[dict] = []          # Flat list of scored article dicts
_categories: Dict[str, list] = {}   # category -> [article dicts]
_last_refresh: str = "Never"
_refresh_lock = threading.Lock()
_refreshing = False
_source_count = 0
_error_message: str = ""


def _ensure_config():
    """Copy config.example.json → config.json if it doesn't exist."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    example_path = os.path.join(os.path.dirname(__file__), "config.example.json")
    if not os.path.exists(config_path) and os.path.exists(example_path):
        shutil.copy2(example_path, config_path)
        logger.info("Created config.json from config.example.json")


def _init_aggregator():
    """Initialise the aggregator singleton."""
    global _aggregator
    _ensure_config()
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    _aggregator = AINewsAggregator(config_path)


def _article_to_dict(article: dict, score: float) -> dict:
    """Convert an (article, score) pair to a JSON-friendly dict."""
    import re

    description = article.get("description", "")
    # Strip HTML
    description = re.sub(r'<script[^>]*?>.*?</script>', '', description, flags=re.DOTALL | re.IGNORECASE)
    description = re.sub(r'<style[^>]*?>.*?</style>', '', description, flags=re.DOTALL | re.IGNORECASE)
    description = re.sub(r'<[^>]+>', '', description)
    description = description.replace('&nbsp;', ' ').replace('&amp;', '&')
    description = description.replace('&lt;', '<').replace('&gt;', '>')
    description = description.replace('&quot;', '"').replace('&#39;', "'")
    description = re.sub(r'\s+', ' ', description).strip()
    if len(description) > 280:
        description = description[:277] + "…"

    pub = article.get("published")
    if pub:
        try:
            pub_str = pub.strftime("%b %d, %Y")
        except Exception:
            pub_str = str(pub)
    else:
        pub_str = ""

    return {
        "title": article.get("title", "Untitled"),
        "link": article.get("link", "#"),
        "source": article.get("source", "Unknown"),
        "score": round(score, 1),
        "date": pub_str,
        "description": description,
    }


def refresh_feeds():
    """Fetch, filter, deduplicate, categorise — update global state."""
    global _articles, _categories, _last_refresh, _refreshing, _source_count, _error_message

    if _refreshing:
        return

    with _refresh_lock:
        _refreshing = True
        _error_message = ""
        try:
            logger.info("Refreshing feeds…")
            raw = _aggregator.fetch_feeds(use_cache=True)
            if not raw:
                _error_message = "No articles could be fetched. Check your internet connection or feed URLs."
                _refreshing = False
                return

            filtered = _aggregator.filter_articles(raw)

            # Dedup
            dedup_cfg = _aggregator.config.get("deduplication", {})
            if dedup_cfg.get("enabled", True):
                filtered = _aggregator.deduplicate_articles(
                    filtered,
                    url_dedup=dedup_cfg.get("url_based", True),
                    title_similarity_threshold=dedup_cfg.get("title_similarity_threshold", 0.85),
                )

            # Build flat list
            flat = [_article_to_dict(a, s) for a, s in filtered]

            # Categorise using the aggregator's method
            categorized_raw = _aggregator.categorize_articles(filtered)
            cats = {}
            for cat, items in categorized_raw.items():
                cats[cat] = [_article_to_dict(a, s) for a, s in items]

            # Count unique sources
            sources = set()
            for a, _ in filtered:
                sources.add(a.get("source", ""))

            _articles = flat
            _categories = cats
            _source_count = len(sources)
            _last_refresh = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
            logger.info(f"Refresh complete — {len(flat)} articles in {len(cats)} categories")

        except Exception as exc:
            logger.exception("Feed refresh failed")
            _error_message = f"Refresh failed: {exc}"
        finally:
            _refreshing = False


def _background_refresh_loop(interval_seconds: int = 1800):
    """Run in a daemon thread — refreshes feeds periodically."""
    while True:
        try:
            refresh_feeds()
        except Exception:
            logger.exception("Background refresh error")
        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main dashboard."""
    # Sort categories by article count (descending)
    sorted_cats = sorted(_categories.items(), key=lambda x: len(x[1]), reverse=True)
    return render_template(
        "index.html",
        sorted_categories=sorted_cats,
        articles=_articles,
        last_refresh=_last_refresh,
        source_count=_source_count,
        total=len(_articles),
        refreshing=_refreshing,
        error=_error_message,
    )


@app.route("/api/feeds")
def api_feeds():
    """JSON API — all articles, optionally filtered by category."""
    cat = request.args.get("category")
    if cat and cat in _categories:
        data = _categories[cat]
    else:
        data = _articles
    return jsonify({
        "articles": data,
        "categories": {k: len(v) for k, v in _categories.items()},
        "total": len(_articles),
        "source_count": _source_count,
        "last_refresh": _last_refresh,
        "refreshing": _refreshing,
        "error": _error_message,
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Trigger a manual feed refresh (async)."""
    if _refreshing:
        return jsonify({"status": "already_refreshing"}), 202

    t = threading.Thread(target=refresh_feeds, daemon=True)
    t.start()
    return jsonify({"status": "refreshing"}), 202


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    _init_aggregator()

    # Initial fetch in background so server starts fast
    t = threading.Thread(target=_background_refresh_loop, daemon=True)
    t.start()

    logger.info("Starting FeedPulse on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)


if __name__ == "__main__":
    main()
