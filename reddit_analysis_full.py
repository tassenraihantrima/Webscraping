import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

df = pd.read_csv('reddit_google_llm_gemini.csv')

df.drop_duplicates()

df['comment'] = df['comment'].fillna('N/A')

df.dropna(subset=['title', 'selftext'], inplace=True)

df['created'] = pd.to_datetime(df['created'], unit='s')

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['post_sentiment'] = df['selftext'].apply(get_sentiment)
df['comment_sentiment'] = df['comment'].apply(get_sentiment)

df['post_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Posts')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

df['comment_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Comments')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

subreddit_post_sentiment = df.groupby('url')['post_sentiment'].mean()

subreddit_post_sentiment.plot(kind='bar', title='Average Post Sentiment by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Average Sentiment')
plt.show()

df['created'] = pd.to_datetime(df['created'])
sentiment_over_time = df.groupby(df['created'].dt.date)['post_sentiment'].mean()

sentiment_over_time.plot(kind='line', title='Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.show()

positive_text = ' '.join(df[df['post_sentiment'] > 0]['selftext'].dropna())
positive_wordcloud = WordCloud(background_color='white').generate(positive_text)

plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Posts')
plt.show()

negative_text = ' '.join(df[df['post_sentiment'] < 0]['selftext'].dropna())
negative_wordcloud = WordCloud(background_color='white').generate(negative_text)

plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Posts')
plt.show()

df.to_csv('cleaned_reddit_data.csv', index=False)

