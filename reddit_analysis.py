
import praw
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('reddit_google_llm_gemini.csv')

df.drop_duplicates(inplace=True)

df['comment'].fillna('N/A', inplace=True)

df.dropna(subset=['title', 'selftext'], inplace=True)

df['created'] = pd.to_datetime(df['created'], unit='s')

top_posts = df.sort_values(by='score', ascending=False).head(10)
print("Top 10 posts:")
print(top_posts[['title', 'score', 'url']])

subreddit_counts = df['url'].groupby(df['url']).count()
print("Post count by subreddit:")
print(subreddit_counts)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['post_sentiment'] = df['selftext'].apply(get_sentiment)
df['comment_sentiment'] = df['comment'].apply(get_sentiment)

print("Average post sentiment:", df['post_sentiment'].mean())
print("Average comment sentiment:", df['comment_sentiment'].mean())

df['post_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Posts')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

df['created'] = pd.to_datetime(df['created'])
post_counts = df.groupby(df['created'].dt.date).size()

post_counts.plot(kind='line', title='Number of Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.show()

all_text = ' '.join(df['selftext'].dropna())
wordcloud = WordCloud(background_color='white').generate(all_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

df.to_csv('cleaned_reddit_data.csv', index=False)

