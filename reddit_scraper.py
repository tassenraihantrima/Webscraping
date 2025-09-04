import praw
import pandas as pd
import prawcore

client_id = 'HFX2IOGvyWk-KYZSVvpAgA'
client_secret = 'AQ_xg-7U881dYXXVT-6_pYZPHESE3w'
user_agent = 'YReddit Scraper v1.0'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, 
user_agent=user_agent)

subreddits = ['MachineLearning', 'OpenAI', 
'LanguageTechnology', 'Google']
search_terms = ['Google LLM', 'Gemini']

data = []

def scrape_reddit_data():
    for subreddit in subreddits:
        print(f'Scraping {subreddit}...')
        for search_term in search_terms:
            try:
                for submission in reddit.subreddit(subreddit).search(search_term, sort='relevance', time_filter='all'):
                    data.append({
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'score': submission.score,
                        'created': submission.created_utc,
                        'url': submission.url
                    })

                    submission.comments.replace_more(limit=None)
                    for comment in submission.comments.list():
                        data.append({
                            'title': submission.title,
                            'selftext': submission.selftext,
                            'score': submission.score,
                            'created': submission.created_utc,
                            'url': submission.url,
                            'comment': comment.body,
                            'comment_score': comment.score
                        })
            except prawcore.exceptions.NotFound:
                print(f"404 Not Found: The subreddit '{subreddit}' or search term '{search_term}' might not exist.")
            except Exception as e:
                print(f"Error while scraping {subreddit} with search term '{search_term}': {e}")

scrape_reddit_data()

df = pd.DataFrame(data)

df.to_csv('reddit_google_llm_gemini.csv', index=False)
print('Scraping complete! Data saved to reddit_google_llm_gemini.csv')

