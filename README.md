# Webscraping

## Reddit Scraping for Google LLM and Gemini

This project scrapes over 27,000 Reddit entries to analyze how people feel about **Google Gemini**. It compares a basic "word-matching" method with a smarter **AI (DistilBERT)** model.

---

## Files

- **reddit_scraper.py**  
  Scrapes posts and comments from selected subreddits using the Reddit API and saves them into a CSV file.

- **reddit_analysis.py**  
  Initial exploratory analysis and basic sentiment testing. Results are stored in the /baseline_results folder.

- **reddit_analysis_full.py**  
  Advanced analysis using a Transformer model on a 2,000-row sample for higher accuracy. Results are stored in the /final_analysis folder.

---

## Visualizations

- **/baseline_results**: Includes the full 27k-row volume timeline, initial sentiment distributions, and general word clouds.
- **/final_analysis**: Includes the AI sentiment charts, timeline for the 2k sample, and positive/negative word clouds.

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```