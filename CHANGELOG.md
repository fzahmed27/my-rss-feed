# Changelog

## [1.1.0] — 2026-01-29 — FeedPulse Web Dashboard

### Added

- **`app.py`** — Flask web server serving a live dashboard on `localhost:5050`
  - Routes: `/` (dashboard), `/api/feeds` (JSON API), `/api/refresh` (manual refresh)
  - Background thread auto-refreshes feeds every 30 minutes
  - Thread-safe global state with refresh lock
  - Auto-creates `config.json` from `config.example.json` if missing

- **`templates/index.html`** — Jinja2 template with Tailwind CSS (CDN)
  - Dark mode default (deep navy `#0f172a` background, blue `#3b82f6` accents)
  - Top nav: FeedPulse logo, search bar, refresh button, dark/light toggle
  - Sidebar: category tabs with article counts (All, Sensors & Hardware, Robotics, People, Economics, ML/AI, CV, Tactile, Other)
  - Stats bar: total articles, active sources, last refresh timestamp
  - Article cards: title (linked), source badge, relevance score pill (color-coded), date, 2-line description
  - Hover effects: subtle lift + blue border glow
  - Responsive: sidebar collapses to horizontal scrollable tabs on mobile
  - Loading state: spinner shown while feeds are being fetched on first load
  - Error state: friendly banner if feeds fail

- **`static/style.css`** — Custom styles
  - Full light mode theme overrides via `html:not(.dark)` selectors
  - Staggered card entrance animations (`fadeSlideIn`)
  - Custom scrollbar styling
  - `prefers-reduced-motion` support

- **`static/app.js`** — Client-side JavaScript
  - Live search: filters articles as the user types (debounced 150ms)
  - Theme toggle: dark/light mode persisted in `localStorage`
  - Category switching: client-side filter + API fetch for fresh data
  - Manual refresh: triggers `/api/refresh`, shows loading overlay, polls until complete
  - Auto-poll on initial load when data isn't ready yet
  - Dynamic article rendering from JSON API

### Changed

- **`requirements.txt`** — Added `flask>=3.0.0`
- **`README.md`** — Added "Web Mode — FeedPulse Dashboard" section with usage instructions, feature list, and API endpoint table; updated project structure diagram

### Unchanged

- `ai_rss_aggregator.py` — No changes; reused as a library by `app.py`
- `feed_cache.py` — No changes
- `config.example.json` — No changes
- `.gitignore` — No changes

### Notes

- The existing CLI mode (`python ai_rss_aggregator.py`) still works exactly as before
- Web mode runs alongside: `python app.py` starts Flask on port 5050
- No git commits or pushes were made; all changes are local
