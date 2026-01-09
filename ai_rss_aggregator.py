#!/usr/bin/env python3
"""
AI RSS Feed Aggregator
Fetches AI/ML news from multiple sources, filters by relevance, and generates daily digests.
"""

import argparse
import datetime
import json
import logging
import os
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

    def generate_html_digest(self, articles: List[Tuple[dict, float]], date_str: str) -> str:
        """
        Generate HTML digest with styling.

        Args:
            articles: List of (article, score) tuples
            date_str: Date string for the digest

        Returns:
            HTML digest
        """
        # Group by source
        by_source = {}
        for article, score in articles:
            source = article['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((article, score))

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI News Digest - {date_str}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #007bff;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .article {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }}
        .article-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .article-title a {{
            color: #333;
            text-decoration: none;
        }}
        .article-title a:hover {{
            color: #007bff;
            text-decoration: underline;
        }}
        .score {{
            display: inline-block;
            background-color: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.9em;
            font-weight: bold;
            margin-right: 10px;
        }}
        .description {{
            color: #666;
            margin-top: 10px;
        }}
        .source-count {{
            color: #666;
            font-weight: normal;
            font-size: 0.9em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI News Digest - {date_str}</h1>
"""

        # Add articles by source
        for source in sorted(by_source.keys()):
            source_articles = by_source[source]
            html += f"""
        <h2>{source} <span class="source-count">({len(source_articles)} articles)</span></h2>
"""

            for article, score in source_articles:
                description = article.get('description', '').replace('\n', ' ').strip()
                if len(description) > 300:
                    description = description[:297] + "..."

                html += f"""
        <div class="article">
            <div class="article-title">
                <span class="score">{score:.1f}</span>
                <a href="{article['link']}" target="_blank">{article['title']}</a>
            </div>
"""
                if description:
                    html += f"""
            <div class="description">{description}</div>
"""
                html += """
        </div>
"""

        html += f"""
        <div class="footer">
            Generated by AI RSS Feed Aggregator on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
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

    def run(self, send_email: bool = True, use_cache: bool = True):
        """
        Run the full aggregation pipeline.

        Args:
            send_email: Whether to send email digest
            use_cache: Whether to use feed cache

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
            use_cache=use_cache
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
