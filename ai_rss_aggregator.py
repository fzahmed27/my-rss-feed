#!/usr/bin/env python3
"""
AI RSS Feed Aggregator
Fetches AI/ML news from multiple sources, filters by relevance, and generates daily digests.
"""

import argparse
import datetime
import difflib
import json
import logging
import os
import re
import smtplib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple

import feedparser
import requests
from dateutil import parser as date_parser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AINewsAggregator:
    """Aggregates and filters AI/ML news from multiple RSS feeds."""

    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize the aggregator with configuration.

        Args:
            config_path: Path to the JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.feeds = self.config['feeds']
        self.keywords = self.config['keywords']
        self.filtering = self.config['filtering']
        self.email_config = self.config.get('email', {})
        self.output_config = self.config.get('output', {'directory': '.', 'filename_prefix': 'ai_digest'})

    def _load_config(self, config_path: str) -> dict:
        """
        Load and validate configuration from JSON file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please copy config.example.json to {config_path} and customize it."
            )

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        # Validate required keys
        required_keys = ['feeds', 'keywords', 'filtering']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        if not config['feeds']:
            raise ValueError("No RSS feeds configured")

        if not config['keywords']:
            raise ValueError("No keywords configured")

        logger.info(f"Loaded config with {len(config['feeds'])} feeds and {len(config['keywords'])} keywords")
        return config

    def fetch_feeds(self, use_cache: bool = True) -> List[dict]:
        """
        Fetch articles from all configured RSS feeds concurrently.

        Args:
            use_cache: Whether to use cached responses (if cache enabled)

        Returns:
            List of articles with metadata
        """
        # Initialize cache if configured
        cache = None
        cache_config = self.config.get('cache', {})
        if cache_config.get('enabled', False) and use_cache:
            from feed_cache import FeedCache
            cache = FeedCache(
                cache_dir=cache_config.get('directory', '.rss_cache'),
                ttl_minutes=cache_config.get('ttl_minutes', 15)
            )

        all_articles = []

        # Fetch all feeds concurrently
        with ThreadPoolExecutor(max_workers=min(10, len(self.feeds))) as executor:
            future_to_source = {
                executor.submit(
                    self._fetch_single_feed,
                    source_name,
                    feed_url,
                    cache
                ): source_name
                for source_name, feed_url in self.feeds.items()
            }

            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.warning(f"Error fetching {source_name}: {e}")

        logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles

    def _fetch_single_feed(
        self,
        source_name: str,
        feed_url: str,
        cache: Optional['FeedCache'] = None
    ) -> List[dict]:
        """
        Fetch a single RSS feed with optional caching.

        Args:
            source_name: Display name for the source
            feed_url: URL of the RSS feed
            cache: Optional cache instance

        Returns:
            List of articles from this feed
        """
        articles = []

        try:
            logger.info(f"Fetching {source_name}...")

            # Build request headers
            headers = {'User-Agent': 'AI-RSS-Aggregator/1.0'}
            content = None
            cache_hit = False

            # Check cache first
            if cache:
                cached_entry = cache.get(feed_url)
                if cached_entry and not cached_entry.is_expired():
                    # Valid cache, use it directly
                    content = cached_entry.content
                    cache_hit = True
                    logger.debug(f"  Cache hit for {source_name}")
                else:
                    # Add conditional headers for stale cache
                    headers.update(cache.get_conditional_headers(feed_url))

            # Fetch from network if needed
            if content is None:
                response = requests.get(
                    feed_url,
                    timeout=10,
                    headers=headers
                )

                if response.status_code == 304:
                    # Not modified - use cached version
                    cached_entry = cache.get(feed_url)
                    if cached_entry:
                        content = cached_entry.content
                        cache_hit = True
                        # Refresh cache timestamp
                        cache.set(
                            feed_url,
                            content,
                            etag=cached_entry.etag,
                            last_modified=cached_entry.last_modified
                        )
                        logger.debug(f"  304 Not Modified for {source_name}")
                else:
                    response.raise_for_status()
                    content = response.content

                    # Update cache
                    if cache:
                        cache.set(
                            feed_url,
                            content,
                            etag=response.headers.get('ETag'),
                            last_modified=response.headers.get('Last-Modified')
                        )

            # Parse the feed
            feed = feedparser.parse(content)

            if feed.bozo and not feed.entries:
                logger.warning(f"Failed to parse feed {source_name}: {feed.bozo_exception}")
                return articles

            # Extract articles
            for entry in feed.entries:
                article = {
                    'source': source_name,
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', ''),
                    'description': entry.get('summary', entry.get('description', '')),
                    'published': self._parse_date(entry),
                    'content': self._extract_content(entry)
                }
                articles.append(article)

            status = "cached" if cache_hit else "fetched"
            logger.info(f"  Found {len(feed.entries)} articles ({status})")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {source_name}: {e}")
        except Exception as e:
            logger.warning(f"Error processing {source_name}: {e}")

        return articles

    def _parse_date(self, entry: dict) -> Optional[datetime.datetime]:
        """
        Parse publication date from feed entry.

        Args:
            entry: Feed entry dictionary

        Returns:
            Parsed datetime or None
        """
        date_fields = ['published', 'updated', 'created']

        for field in date_fields:
            if field in entry:
                try:
                    return date_parser.parse(entry[field])
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_content(self, entry: dict) -> str:
        """
        Extract text content from feed entry.

        Args:
            entry: Feed entry dictionary

        Returns:
            Extracted text content
        """
        # Try different content fields
        if 'content' in entry and entry.content:
            return ' '.join([c.get('value', '') for c in entry.content])

        return entry.get('summary', entry.get('description', ''))

    def _strip_html_tags(self, html_text: str) -> str:
        """
        Strip HTML tags from text while preserving readable content.

        Args:
            html_text: Text potentially containing HTML

        Returns:
            Plain text with HTML tags removed
        """
        if not html_text:
            return ''

        # Remove <script> and <style> tags and their content
        html_text = re.sub(r'<script[^>]*?>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        html_text = re.sub(r'<style[^>]*?>.*?</style>', '', html_text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        html_text = re.sub(r'<!--.*?-->', '', html_text, flags=re.DOTALL)

        # Replace <br> and <p> tags with spaces to preserve readability
        html_text = re.sub(r'<br\s*/?>|</p>', ' ', html_text, flags=re.IGNORECASE)

        # Remove all other HTML tags
        html_text = re.sub(r'<[^>]+>', '', html_text)

        # Decode common HTML entities
        html_text = html_text.replace('&nbsp;', ' ')
        html_text = html_text.replace('&amp;', '&')
        html_text = html_text.replace('&lt;', '<')
        html_text = html_text.replace('&gt;', '>')
        html_text = html_text.replace('&quot;', '"')
        html_text = html_text.replace('&#39;', "'")

        # Collapse multiple whitespace into single space
        html_text = re.sub(r'\s+', ' ', html_text)

        return html_text.strip()

    def score_article(self, article: dict) -> float:
        """
        Calculate relevance score for an article based on keywords.

        Scoring weights:
        - Title matches: 3x keyword weight
        - Description matches: 2x keyword weight
        - Content matches: 1x keyword weight

        Args:
            article: Article dictionary

        Returns:
            Relevance score
        """
        score = 0.0

        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()

        for keyword, weight in self.keywords.items():
            keyword_lower = keyword.lower()

            # Title matches (3x weight)
            if keyword_lower in title:
                score += weight * 3.0

            # Description matches (2x weight)
            if keyword_lower in description:
                score += weight * 2.0

            # Content matches (1x weight)
            if keyword_lower in content:
                score += weight * 1.0

        return score

    def filter_articles(
        self,
        articles: List[dict],
        min_score: Optional[float] = None,
        days_back: Optional[int] = None
    ) -> List[Tuple[dict, float]]:
        """
        Filter articles by relevance score and date.

        Args:
            articles: List of articles
            min_score: Minimum relevance score (uses config if not provided)
            days_back: Number of days to look back (uses config if not provided)

        Returns:
            List of (article, score) tuples sorted by score (highest first)
        """
        if min_score is None:
            min_score = self.filtering['min_score']

        if days_back is None:
            days_back = self.filtering['days_back']

        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)

        filtered = []

        for article in articles:
            score = self.score_article(article)

            # Check score threshold
            if score < min_score:
                continue

            # Check date threshold
            pub_date = article.get('published')
            if pub_date:
                # Make timezone-aware if needed
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=datetime.timezone.utc)

                if pub_date < cutoff_date:
                    continue

            filtered.append((article, score))

        # Sort by score (highest first)
        filtered.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Filtered to {len(filtered)} relevant articles (min_score={min_score}, days_back={days_back})")
        return filtered

    def deduplicate_articles(
        self,
        articles: List[Tuple[dict, float]],
        url_dedup: bool = True,
        title_similarity_threshold: float = 0.85
    ) -> List[Tuple[dict, float]]:
        """
        Remove duplicate articles using URL matching and title similarity.

        Args:
            articles: List of (article, score) tuples
            url_dedup: Enable URL-based deduplication
            title_similarity_threshold: Minimum similarity ratio for title matching (0.0-1.0)

        Returns:
            Deduplicated list of (article, score) tuples
        """
        if not articles:
            return articles

        deduplicated = []
        seen_urls = set()
        seen_titles = []

        for article, score in articles:
            url = article.get('link', '').strip()
            title = article.get('title', '').strip().lower()

            # Skip if no URL or title
            if not url and not title:
                continue

            # URL-based deduplication (exact match)
            if url_dedup and url:
                if url in seen_urls:
                    logger.debug(f"Skipping duplicate URL: {article['title'][:50]}...")
                    continue
                seen_urls.add(url)

            # Title similarity deduplication
            is_duplicate = False
            if title:
                for seen_title, seen_score in seen_titles:
                    similarity = difflib.SequenceMatcher(None, title, seen_title).ratio()
                    if similarity >= title_similarity_threshold:
                        # Keep the one with higher score
                        if score > seen_score:
                            # Remove the previous one and add this one
                            deduplicated = [
                                (a, s) for a, s in deduplicated
                                if a.get('title', '').strip().lower() != seen_title
                            ]
                            seen_titles.remove((seen_title, seen_score))
                            logger.debug(
                                f"Replacing duplicate (similarity: {similarity:.2f}): "
                                f"{article['title'][:50]}..."
                            )
                            break
                        else:
                            logger.debug(
                                f"Skipping duplicate (similarity: {similarity:.2f}): "
                                f"{article['title'][:50]}..."
                            )
                            is_duplicate = True
                            break

            if not is_duplicate:
                deduplicated.append((article, score))
                if title:
                    seen_titles.append((title, score))

        removed_count = len(articles) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate articles")

        return deduplicated

    def generate_text_digest(self, articles: List[Tuple[dict, float]], date_str: str) -> str:
        """
        Generate plain text digest.

        Args:
            articles: List of (article, score) tuples
            date_str: Date string for the digest

        Returns:
            Plain text digest
        """
        lines = []
        lines.append(f"AI News Digest - {date_str}")
        lines.append("=" * 60)
        lines.append("")

        # Group by source
        by_source = {}
        for article, score in articles:
            source = article['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((article, score))

        # Generate output for each source
        for source in sorted(by_source.keys()):
            lines.append(f"{source} ({len(by_source[source])} articles)")
            lines.append("-" * 60)
            lines.append("")

            for article, score in by_source[source]:
                lines.append(f"[Score: {score:.1f}] {article['title']}")
                lines.append(f"Link: {article['link']}")

                # Add description snippet
                description = article.get('description', '')
                if description:
                    # Clean and truncate
                    description = description.replace('\n', ' ').strip()
                    if len(description) > 200:
                        description = description[:197] + "..."
                    lines.append(description)

                lines.append("")

            lines.append("")

        return '\n'.join(lines)

    def generate_markdown_digest(self, articles: List[Tuple[dict, float]], date_str: str) -> str:
        """
        Generate markdown digest.

        Args:
            articles: List of (article, score) tuples
            date_str: Date string for the digest

        Returns:
            Markdown digest
        """
        lines = []
        lines.append(f"# AI News Digest - {date_str}")
        lines.append("")

        # Group by source
        by_source = {}
        for article, score in articles:
            source = article['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((article, score))

        # Generate output for each source
        for source in sorted(by_source.keys()):
            lines.append(f"## {source} ({len(by_source[source])} articles)")
            lines.append("")

            for article, score in by_source[source]:
                lines.append(f"### [{article['title']}]({article['link']})")
                lines.append(f"**Relevance Score:** {score:.1f}")
                lines.append("")

                # Add description
                description = article.get('description', '')
                if description:
                    description = description.replace('\n', ' ').strip()
                    if len(description) > 300:
                        description = description[:297] + "..."
                    lines.append(description)
                    lines.append("")

            lines.append("")

        return '\n'.join(lines)

    def categorize_articles(self, articles: List[Tuple[dict, float]]) -> Dict[str, List[Tuple[dict, float]]]:
        """
        Categorize articles into topic groups based on keywords and source.
        Each article is assigned to only ONE category - its best match.

        Args:
            articles: List of (article, score) tuples

        Returns:
            Dictionary mapping category names to lists of (article, score) tuples
        """
        # Define sources that belong to "People" category
        people_sources = {
            'BAIR Blog', 'Andrej Karpathy', 'Robot Brains Podcast',
            'Marginal Revolution', 'Paul Krugman', 'Noahpinion',
            'Abnormal Returns', 'The Reformed Broker'
        }

        # Define keyword categories with priority order
        # Higher priority = more specific topics
        categories = {
            'People': [],  # Special category for individual bloggers/thought leaders
            'Tactile & Haptics': ['tactile', 'haptic', 'haptics', 'touch sensing', 'force sensing'],
            'Computer Vision': ['computer vision', 'image recognition', 'object detection', 'semantic segmentation',
                              'depth estimation', 'cnn', 'convolutional'],
            'Robotics & Manipulation': ['robotics', 'robot', 'grasp', 'grasping', 'manipulation', 'dexterous', 'autonomous'],
            'Machine Learning & AI': ['machine learning', 'deep learning', 'neural network', 'pytorch'],
            'Sensors & Hardware': ['sensor', 'sensors', 'embedded', 'edge', 'inference', 'model compression',
                                  'quantization', 'optimization'],
            'Economics & Markets': ['economics', 'economy', 'inflation', 'monetary policy', 'fiscal policy',
                                   'GDP', 'recession', 'stock', 'investing', 'portfolio', 'trading',
                                   'equity', 'bonds', 'ETF', 'valuation', 'dividend', 'bull market', 'bear market'],
            'Other': []  # Catch-all for articles that don't fit other categories
        }

        categorized = {cat: [] for cat in categories.keys()}

        for article, score in articles:
            source = article.get('source', '')

            # Check if article is from a "People" source first (highest priority)
            if source in people_sources:
                categorized['People'].append((article, score))
                continue

            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = article.get('content', '').lower()
            full_text = f"{title} {description} {content}"

            # Find the BEST matching category (first match in priority order)
            # This prevents duplicate articles across tabs
            best_category = None
            max_keyword_matches = 0

            for category, keywords in categories.items():
                if category in ['Other', 'People']:
                    continue

                # Count keyword matches for this category
                keyword_matches = 0
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        keyword_matches += 1

                # Use the category with the most keyword matches
                if keyword_matches > max_keyword_matches:
                    max_keyword_matches = keyword_matches
                    best_category = category

            # Assign to best matching category
            if best_category:
                categorized[best_category].append((article, score))
            else:
                # No match, put in "Other"
                categorized['Other'].append((article, score))

        # Remove empty categories
        categorized = {cat: arts for cat, arts in categorized.items() if arts}

        return categorized

    def generate_html_digest(self, articles: List[Tuple[dict, float]], date_str: str) -> str:
        """
        Generate HTML digest with tabbed topic categories.

        Args:
            articles: List of (article, score) tuples
            date_str: Date string for the digest

        Returns:
            HTML digest with tabs
        """
        # Categorize articles by topic
        categorized = self.categorize_articles(articles)

        # Sort categories by article count (descending)
        sorted_categories = sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI News Digest - {date_str}</title>
    <style>
        /* Design Tokens - CSS Custom Properties */
        :root {{
            /* Colors - Light Mode */
            --color-primary: #0066cc;
            --color-primary-hover: #0052a3;
            --color-primary-light: #e6f2ff;

            --color-surface: #ffffff;
            --color-surface-secondary: #f8f9fa;
            --color-surface-tertiary: #f0f2f5;

            --color-text-primary: #1a1a1a;
            --color-text-secondary: #4a5568;
            --color-text-tertiary: #718096;

            --color-border: #e2e8f0;
            --color-border-focus: #0066cc;

            --color-success: #059669;
            --color-info: #64748b;

            /* Spacing Scale (8px base) */
            --space-1: 0.25rem;  /* 4px */
            --space-2: 0.5rem;   /* 8px */
            --space-3: 0.75rem;  /* 12px */
            --space-4: 1rem;     /* 16px */
            --space-5: 1.25rem;  /* 20px */
            --space-6: 1.5rem;   /* 24px */
            --space-8: 2rem;     /* 32px */
            --space-10: 2.5rem;  /* 40px */

            /* Typography Scale */
            --font-size-sm: 0.875rem;   /* 14px */
            --font-size-base: 1rem;     /* 16px */
            --font-size-md: 1.125rem;   /* 18px */
            --font-size-lg: 1.25rem;    /* 20px */
            --font-size-xl: 1.5rem;     /* 24px */
            --font-size-2xl: 2rem;      /* 32px */

            --font-weight-normal: 400;
            --font-weight-medium: 500;
            --font-weight-semibold: 600;
            --font-weight-bold: 700;

            --line-height-tight: 1.4;
            --line-height-normal: 1.6;
            --line-height-relaxed: 1.8;

            /* Border Radius */
            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --radius-full: 9999px;

            /* Shadows */
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.15);

            /* Transitions */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* Dark Mode Support */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --color-primary: #3b82f6;
                --color-primary-hover: #60a5fa;
                --color-primary-light: #1e3a5f;

                --color-surface: #1a1a1a;
                --color-surface-secondary: #2d2d2d;
                --color-surface-tertiary: #3a3a3a;

                --color-text-primary: #f5f5f5;
                --color-text-secondary: #d1d5db;
                --color-text-tertiary: #9ca3af;

                --color-border: #404040;
                --color-border-focus: #3b82f6;

                --color-success: #10b981;
                --color-info: #94a3b8;
            }}
        }}

        /* Base Styles */
        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: var(--line-height-normal);
            margin: 0;
            padding: var(--space-4);
            background-color: var(--color-surface-tertiary);
            color: var(--color-text-primary);
            font-size: var(--font-size-base);
        }}

        /* Skip to Main Content */
        .skip-link {{
            position: absolute;
            top: -40px;
            left: 0;
            background: var(--color-primary);
            color: white;
            padding: var(--space-2) var(--space-4);
            text-decoration: none;
            border-radius: var(--radius-sm);
            z-index: 100;
        }}

        .skip-link:focus {{
            top: var(--space-2);
            left: var(--space-2);
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: var(--color-surface);
            padding: var(--space-6);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-md);
        }}

        h1 {{
            font-size: var(--font-size-2xl);
            font-weight: var(--font-weight-bold);
            color: var(--color-text-primary);
            margin: 0 0 var(--space-4) 0;
            padding-bottom: var(--space-4);
            border-bottom: 3px solid var(--color-primary);
            line-height: var(--line-height-tight);
        }}

        .summary {{
            color: var(--color-text-secondary);
            font-size: var(--font-size-sm);
            margin-bottom: var(--space-6);
            padding: var(--space-4);
            background-color: var(--color-surface-secondary);
            border-radius: var(--radius-md);
            border-left: 4px solid var(--color-primary);
        }}

        /* Tabs - Mobile First with Horizontal Scroll */
        .tabs {{
            display: flex;
            gap: var(--space-2);
            margin-bottom: var(--space-4);
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
            scrollbar-color: var(--color-border) transparent;
            padding-bottom: var(--space-2);
            border-bottom: 2px solid var(--color-border);
        }}

        .tabs::-webkit-scrollbar {{
            height: 6px;
        }}

        .tabs::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .tabs::-webkit-scrollbar-thumb {{
            background-color: var(--color-border);
            border-radius: var(--radius-full);
        }}

        .tab {{
            flex-shrink: 0;
            min-width: 120px;
            padding: var(--space-3) var(--space-5);
            background-color: var(--color-surface-secondary);
            border: 2px solid transparent;
            cursor: pointer;
            font-size: var(--font-size-sm);
            font-weight: var(--font-weight-medium);
            color: var(--color-text-secondary);
            border-radius: var(--radius-md);
            transition: all var(--transition-base);
            white-space: nowrap;
            text-align: center;
        }}

        .tab:hover {{
            background-color: var(--color-primary-light);
            color: var(--color-primary);
            transform: translateY(-2px);
        }}

        .tab:focus {{
            outline: 2px solid var(--color-border-focus);
            outline-offset: 2px;
        }}

        .tab.active {{
            background-color: var(--color-primary);
            color: white;
            border-color: var(--color-primary);
            font-weight: var(--font-weight-semibold);
        }}

        .tab-content {{
            display: none;
            padding: var(--space-6) 0;
            animation: fadeIn var(--transition-base);
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(8px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        h2.category-header {{
            font-size: var(--font-size-xl);
            font-weight: var(--font-weight-semibold);
            color: var(--color-primary);
            margin: 0 0 var(--space-6) 0;
            padding-bottom: var(--space-3);
            border-bottom: 2px solid var(--color-border);
        }}

        /* Article Cards */
        article.article {{
            margin-bottom: var(--space-5);
            padding: var(--space-5);
            background-color: var(--color-surface-secondary);
            border-left: 4px solid var(--color-primary);
            border-radius: var(--radius-md);
            transition: all var(--transition-base);
        }}

        article.article:hover {{
            box-shadow: var(--shadow-md);
            background-color: var(--color-surface);
            transform: translateX(4px);
        }}

        .article-title {{
            font-size: var(--font-size-md);
            font-weight: var(--font-weight-semibold);
            margin-bottom: var(--space-3);
            line-height: var(--line-height-tight);
        }}

        .article-title a {{
            color: var(--color-text-primary);
            text-decoration: none;
            transition: color var(--transition-fast);
        }}

        .article-title a:hover {{
            color: var(--color-primary);
            text-decoration: underline;
        }}

        .article-title a:focus {{
            outline: 2px solid var(--color-border-focus);
            outline-offset: 2px;
            border-radius: var(--radius-sm);
        }}

        .article-meta {{
            display: flex;
            gap: var(--space-3);
            align-items: center;
            margin-bottom: var(--space-3);
            flex-wrap: wrap;
        }}

        .score {{
            display: inline-flex;
            align-items: center;
            background-color: var(--color-success);
            color: white;
            padding: var(--space-1) var(--space-3);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            font-weight: var(--font-weight-semibold);
            line-height: 1;
        }}

        .source-badge {{
            display: inline-flex;
            align-items: center;
            background-color: var(--color-info);
            color: white;
            padding: var(--space-1) var(--space-3);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            font-weight: var(--font-weight-medium);
            line-height: 1;
        }}

        .description {{
            color: var(--color-text-secondary);
            font-size: var(--font-size-sm);
            line-height: var(--line-height-relaxed);
            overflow: hidden;
        }}

        /* Constrain any images or media in descriptions */
        .description img,
        .description video,
        .description iframe {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: var(--space-3) 0;
            border-radius: var(--radius-sm);
        }}

        .footer {{
            margin-top: var(--space-10);
            padding-top: var(--space-6);
            border-top: 1px solid var(--color-border);
            color: var(--color-text-tertiary);
            font-size: var(--font-size-sm);
            text-align: center;
        }}

        .count-badge {{
            display: inline-block;
            background-color: var(--color-primary);
            color: white;
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            font-weight: var(--font-weight-semibold);
            line-height: 1;
            margin-left: var(--space-1);
        }}

        /* Tablet Breakpoint (768px+) */
        @media (min-width: 768px) {{
            body {{
                padding: var(--space-6);
            }}

            .container {{
                padding: var(--space-8);
            }}

            .tabs {{
                flex-wrap: wrap;
                overflow-x: visible;
            }}

            .tab {{
                min-width: auto;
            }}

            h1 {{
                font-size: clamp(var(--font-size-xl), 4vw, var(--font-size-2xl));
            }}
        }}

        /* Desktop Breakpoint (1024px+) */
        @media (min-width: 1024px) {{
            body {{
                padding: var(--space-8);
            }}

            .container {{
                padding: var(--space-10);
            }}
        }}

        /* Reduced Motion Support */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}

        /* Print Styles */
        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
                padding: 0;
            }}

            .tabs {{
                display: none;
            }}

            .tab-content {{
                display: block !important;
                page-break-before: always;
            }}

            article.article {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>

    <div class="container">
        <h1>🤖 AI News Digest - {date_str}</h1>
        <div class="summary" role="status" aria-live="polite">
            📊 Total: {len(articles)} articles across {len(categorized)} topic categories
        </div>

        <div class="tabs" role="tablist" aria-label="Article categories">
"""

        # Generate tab buttons with ARIA
        for idx, (category, cat_articles) in enumerate(sorted_categories):
            is_active = idx == 0
            safe_category = category.replace(' ', '_').replace('&', 'and')
            html += f"""            <button
                class="tab{"  active" if is_active else ""}"
                role="tab"
                aria-selected="{"true" if is_active else "false"}"
                aria-controls="{safe_category}"
                id="{safe_category}-tab"
                tabindex="{"0" if is_active else "-1"}"
                onclick="openTab(event, '{safe_category}')"
            >
                {category} <span class="count-badge" aria-label="{len(cat_articles)} articles">{len(cat_articles)}</span>
            </button>
"""

        html += """        </div>
"""

        # Generate tab contents with ARIA and semantic HTML
        for idx, (category, cat_articles) in enumerate(sorted_categories):
            is_active = idx == 0
            safe_category = category.replace(' ', '_').replace('&', 'and')

            html += f"""
        <div
            id="{safe_category}"
            class="tab-content{"  active" if is_active else ""}"
            role="tabpanel"
            aria-labelledby="{safe_category}-tab"
            tabindex="0"
        >
            <h2 class="category-header">{category} <span class="count-badge">{len(cat_articles)}</span></h2>
"""

            for article, score in cat_articles:
                # Strip HTML tags from description and clean up
                raw_description = article.get('description', '')
                description = self._strip_html_tags(raw_description)
                description = description.replace('\n', ' ').strip()
                if len(description) > 300:
                    description = description[:297] + "..."

                source = article.get('source', 'Unknown')
                # Escape HTML entities in title for safety
                title = article.get('title', 'Untitled').replace('<', '&lt;').replace('>', '&gt;')

                html += f"""
            <article class="article">
                <h3 class="article-title">
                    <a href="{article['link']}" target="_blank" rel="noopener noreferrer">{title}</a>
                </h3>
                <div class="article-meta">
                    <span class="score" aria-label="Relevance score {score:.1f}">⭐ {score:.1f}</span>
                    <span class="source-badge" aria-label="Source: {source}">📰 {source}</span>
                </div>
"""
                if description:
                    html += f"""                <p class="description">{description}</p>
"""
                html += """            </article>
"""

            html += """        </div>
"""

        html += f"""
        <footer class="footer">
            Generated by AI RSS Feed Aggregator on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>

    <script>
        function openTab(evt, tabName) {{
            // Get all tabs and tab contents
            var tabContents = document.querySelectorAll('[role="tabpanel"]');
            var tabs = document.querySelectorAll('[role="tab"]');

            // Hide all tab contents and deactivate all tabs
            tabContents.forEach(function(content) {{
                content.classList.remove('active');
            }});

            tabs.forEach(function(tab) {{
                tab.classList.remove('active');
                tab.setAttribute('aria-selected', 'false');
                tab.setAttribute('tabindex', '-1');
            }});

            // Show selected tab content and activate tab
            var selectedPanel = document.getElementById(tabName);
            if (selectedPanel) {{
                selectedPanel.classList.add('active');
                selectedPanel.focus();
            }}

            if (evt && evt.currentTarget) {{
                evt.currentTarget.classList.add('active');
                evt.currentTarget.setAttribute('aria-selected', 'true');
                evt.currentTarget.setAttribute('tabindex', '0');
            }}
        }}

        // Keyboard navigation for tabs
        document.addEventListener('DOMContentLoaded', function() {{
            var tabs = document.querySelectorAll('[role="tab"]');

            tabs.forEach(function(tab, index) {{
                tab.addEventListener('keydown', function(e) {{
                    var currentIndex = Array.from(tabs).indexOf(e.target);
                    var nextIndex;

                    // Arrow key navigation
                    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                        e.preventDefault();
                        nextIndex = (currentIndex + 1) % tabs.length;
                        tabs[nextIndex].focus();
                        tabs[nextIndex].click();
                    }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                        e.preventDefault();
                        nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
                        tabs[nextIndex].focus();
                        tabs[nextIndex].click();
                    }} else if (e.key === 'Home') {{
                        e.preventDefault();
                        tabs[0].focus();
                        tabs[0].click();
                    }} else if (e.key === 'End') {{
                        e.preventDefault();
                        tabs[tabs.length - 1].focus();
                        tabs[tabs.length - 1].click();
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>
"""

        return html

    def send_email(self, html_content: str, subject: str):
        """
        Send digest via email.

        Args:
            html_content: HTML content to send
            subject: Email subject
        """
        if not self.email_config.get('enabled', False):
            logger.info("Email disabled in config, skipping")
            return

        try:
            sender = self.email_config['sender_email']
            password = self.email_config['sender_password']
            recipient = self.email_config['recipient_email']
            smtp_server = self.email_config['smtp_server']
            smtp_port = self.email_config['smtp_port']

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = recipient

            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            logger.info(f"Sending email to {recipient}...")
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())

            logger.info("Email sent successfully")

        except KeyError as e:
            logger.error(f"Missing email configuration: {e}")
        except smtplib.SMTPException as e:
            logger.error(f"Failed to send email: {e}")
        except Exception as e:
            logger.error(f"Error sending email: {e}")

    def run(self, send_email: bool = True, use_cache: bool = True, enable_dedup: bool = True):
        """
        Run the full aggregation pipeline.

        Args:
            send_email: Whether to send email digest
            use_cache: Whether to use feed cache
            enable_dedup: Whether to enable deduplication

        Returns:
            Tuple of (articles_count, filtered_count)
        """
        logger.info("Starting AI News Aggregator")

        # Fetch articles (with caching if enabled)
        articles = self.fetch_feeds(use_cache=use_cache)

        if not articles:
            logger.warning("No articles fetched")
            return 0, 0

        # Filter articles
        filtered = self.filter_articles(articles)

        if not filtered:
            logger.warning("No articles passed the filter")
            if not self.email_config.get('send_if_empty', False):
                logger.info("Skipping output generation for empty results")
                return len(articles), 0

        # Deduplicate articles
        dedup_config = self.config.get('deduplication', {})
        if enable_dedup and dedup_config.get('enabled', True):
            filtered = self.deduplicate_articles(
                filtered,
                url_dedup=dedup_config.get('url_based', True),
                title_similarity_threshold=dedup_config.get('title_similarity_threshold', 0.85)
            )

        # Generate outputs
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        date_stamp = datetime.datetime.now().strftime('%Y%m%d')

        output_dir = self.output_config['directory']
        prefix = self.output_config['filename_prefix']

        # Ensure output directory exists
        if output_dir and output_dir != '.':
            os.makedirs(output_dir, exist_ok=True)

        # Generate text digest
        text_content = self.generate_text_digest(filtered, date_str)
        text_file = os.path.join(output_dir, f"{prefix}_{date_stamp}.txt")
        with open(text_file, 'w') as f:
            f.write(text_content)
        logger.info(f"Generated text digest: {text_file}")

        # Generate markdown digest
        md_content = self.generate_markdown_digest(filtered, date_str)
        md_file = os.path.join(output_dir, f"{prefix}_{date_stamp}.md")
        with open(md_file, 'w') as f:
            f.write(md_content)
        logger.info(f"Generated markdown digest: {md_file}")

        # Generate HTML digest
        html_content = self.generate_html_digest(filtered, date_str)
        html_file = os.path.join(output_dir, f"{prefix}_{date_stamp}.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Generated HTML digest: {html_file}")

        # Send email if requested
        if send_email:
            subject = f"AI News Digest - {date_str}"
            self.send_email(html_content, subject)

        logger.info(f"Done! Processed {len(articles)} articles, {len(filtered)} passed filter")
        return len(articles), len(filtered)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI RSS Feed Aggregator - Fetch and filter AI/ML news'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to config file (default: config.json)'
    )
    parser.add_argument(
        '--no-email',
        action='store_true',
        help='Skip sending email digest'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable feed caching (always fetch fresh)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the feed cache before running'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh all feeds (ignore cache)'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable article deduplication'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize aggregator
        aggregator = AINewsAggregator(args.config)

        # Handle cache clearing
        if args.clear_cache:
            cache_config = aggregator.config.get('cache', {})
            if cache_config.get('enabled', False):
                from feed_cache import FeedCache
                cache = FeedCache(cache_config.get('directory', '.rss_cache'))
                cleared = cache.clear()
                logger.info(f"Cleared {cleared} cached entries")

        # Determine cache usage
        use_cache = not (args.no_cache or args.force_refresh)

        # Run aggregator
        total, filtered = aggregator.run(
            send_email=not args.no_email,
            use_cache=use_cache,
            enable_dedup=not args.no_dedup
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"AI News Digest Summary")
        print(f"{'='*60}")
        print(f"Total articles fetched: {total}")
        print(f"Relevant articles: {filtered}")
        print(f"Output files generated in: {aggregator.output_config['directory']}")
        print(f"{'='*60}\n")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
