# Webscraping

## Reddit Scraping for Google LLM and Gemini

This project scrapes Reddit posts and comments from subreddits related to **Google LLM** and **Gemini**, then performs sentiment analysis and visualization.

---

## Files

- **reddit_scraper.py**  
  Scrapes posts and comments from selected subreddits using the Reddit API (PRAW) and saves them into a CSV file.

- **reddit_analysis.py**  
  Basic analysis of the scraped data: top posts, counts, sentiment distribution, and word cloud.

- **reddit_analysis_full.py**  
  Extended analysis with additional visualizations such as sentiment over time, average sentiment by subreddit, and positive/negative word clouds.

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
