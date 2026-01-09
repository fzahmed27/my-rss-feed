# AI RSS Feed Aggregator

A Python script that fetches AI/ML news from multiple sources, filters for topics relevant to your work, and generates daily digests.

## Features

- Fetches from 9+ major AI/ML RSS feeds (OpenAI, DeepMind, PyTorch, Hugging Face, etc.)
- Filters content based on relevance to computer vision, PyTorch, sensors, and your other interests
- Scores articles by relevance (higher score = more relevant)
- Generates digests in 3 formats: text, markdown, and HTML
- Optional email delivery with SMTP support
- Customizable keywords and weights via JSON config
- Robust error handling and logging

## Setup

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create your configuration file:
```bash
cp config.example.json config.json
```

4. Edit `config.json` to customize:
   - RSS feed sources
   - Keywords and weights
   - Filtering thresholds
   - Email settings (optional)

5. Run the script:
```bash
python ai_rss_aggregator.py
```

## Output

The script generates three files:
- `ai_digest_YYYYMMDD.txt` - Plain text digest
- `ai_digest_YYYYMMDD.md` - Markdown digest (great for note-taking apps)
- `ai_digest_YYYYMMDD.html` - HTML digest (open in browser for nice formatting)

## Configuration

All configuration is managed through `config.json`:

### RSS Feeds

```json
"feeds": {
  "OpenAI": "https://openai.com/blog/rss.xml",
  "PyTorch Blog": "https://pytorch.org/blog/feed.xml"
}
```

### Keywords and Weights

Higher weights = higher relevance scores. The scoring algorithm:
- Title matches: 3x keyword weight
- Description matches: 2x keyword weight
- Content matches: 1x keyword weight

```json
"keywords": {
  "computer vision": 3.0,
  "pytorch": 2.5,
  "sensors": 2.0,
  "machine learning": 1.5
}
```

### Filtering Thresholds

```json
"filtering": {
  "min_score": 1.5,
  "days_back": 30
}
```

- `min_score`: Minimum relevance score to include an article
- `days_back`: Only include articles from the last N days

### Email Configuration (Optional)

To enable email delivery:

```json
"email": {
  "enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 465,
  "sender_email": "your_email@gmail.com",
  "sender_password": "your_app_password",
  "recipient_email": "your_email@gmail.com",
  "send_if_empty": false
}
```

**For Gmail:**
1. Enable 2-factor authentication
2. Generate an "App Password" at https://myaccount.google.com/apppasswords
3. Use the app password in the config (not your regular password)

## Command Line Options

```bash
# Use a custom config file
python ai_rss_aggregator.py --config my_config.json

# Skip sending email
python ai_rss_aggregator.py --no-email

# Enable verbose logging
python ai_rss_aggregator.py --verbose
```

## Automation

### Daily Digest via Cron (Linux/Mac)

Run the script daily at 8 AM:

```bash
crontab -e
# Add this line (adjust paths):
0 8 * * * cd /path/to/rss-feed && /path/to/venv/bin/python ai_rss_aggregator.py
```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (daily at your preferred time)
4. Action: Start a program
5. Program: `C:\path\to\venv\Scripts\python.exe`
6. Arguments: `C:\path\to\ai_rss_aggregator.py`
7. Start in: `C:\path\to\rss-feed`

## Example Output

```
AI News Digest - 2026-01-08
============================================================

PyTorch Blog (3 articles)
------------------------------------------------------------

[Score: 24.5] Deploying Smarter: Hardware-Software Co-design in PyTorch
Link: https://pytorch.org/blog/...
If you want powerful on-device AI that doesn't blow your memory budget...

[Score: 20.5] PyTorch 2.9: FlexAttention Optimization Practice
Link: https://pytorch.org/blog/...
The most recent LLM serving frameworks increasingly adopt attention variants...

Microsoft Research (2 articles)
------------------------------------------------------------

[Score: 3.0] Agent Lightning: Adding RL to AI agents
Link: https://www.microsoft.com/...
By decoupling how agents work from how they're trained...
```

## Tips for Reducing Social Media Time

1. **Set a schedule**: Run this once daily (morning is good)
2. **Time-box reading**: Give yourself 10-15 minutes to read the digest
3. **Trust the filter**: If something major happens, it'll be in your digest
4. **No FOMO**: Real breakthroughs take time to matter; you don't need to know instantly
5. **Track your time**: See how much time you save vs scrolling social media

## Troubleshooting

### RSS Feed 404 Errors

Some RSS feed URLs may change over time. If you see 404 errors in the logs, check if the source has moved their RSS feed:

- Search for "[Source Name] RSS feed" to find the current URL
- Update the URL in your `config.json`
- Or remove feeds that are no longer available

### Email Not Sending

- Verify SMTP settings are correct
- For Gmail, make sure you're using an App Password, not your regular password
- Check that "enabled" is set to `true` in email config
- Run with `--verbose` to see detailed error messages

### No Articles Pass Filter

If you're getting 0 filtered articles:
- Lower `min_score` threshold (try 1.0 instead of 1.5)
- Increase `days_back` (try 60 or 90 days)
- Add more keywords or adjust weights
- Check logs to see if articles are being fetched at all

## Project Structure

```
rss-feed/
├── ai_rss_aggregator.py    # Main script
├── config.json              # Your configuration (not in git)
├── config.example.json      # Example configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── venv/                   # Virtual environment (not in git)
└── ai_digest_YYYYMMDD.*    # Generated digests
```

## License

MIT License - feel free to modify and use as you wish.

## Contributing

This is a personal tool, but suggestions and improvements are welcome! Open an issue or submit a pull request.
