import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# Load the CSV file (after scraping)
df = pd.read_csv('reddit_google_llm_gemini.csv')

# Data Cleaning and Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing comment text with 'N/A'
df['comment'].fillna('N/A', inplace=True)

# Drop rows where title or selftext is missing
df.dropna(subset=['title', 'selftext'], inplace=True)

# Convert Unix timestamp to readable date
df['created'] = pd.to_datetime(df['created'], unit='s')

# Sentiment Analysis
# Define a function to calculate sentiment using TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the function to posts and comments
df['post_sentiment'] = df['selftext'].apply(get_sentiment)
df['comment_sentiment'] = df['comment'].apply(get_sentiment)

# Sentiment Distribution for Posts and Comments
# Plot sentiment distribution for posts
df['post_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Posts')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Plot sentiment distribution for comments
df['comment_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Comments')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Sentiment by Subreddit
# Average sentiment by subreddit for posts
subreddit_post_sentiment = df.groupby('url')['post_sentiment'].mean()

# Plotting average sentiment by subreddit
subreddit_post_sentiment.plot(kind='bar', title='Average Post Sentiment by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Average Sentiment')
plt.show()

# Sentiment Over Time
# Group by date and calculate the average sentiment for posts
df['created'] = pd.to_datetime(df['created'])
sentiment_over_time = df.groupby(df['created'].dt.date)['post_sentiment'].mean()

# Plotting sentiment over time
sentiment_over_time.plot(kind='line', title='Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.show()

# Word Cloud for Positive and Negative Sentiment
positive_text = ' '.join(df[df['post_sentiment'] > 0]['selftext'].dropna())
positive_wordcloud = WordCloud(background_color='white').generate(positive_text)

plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Posts')
plt.show()

# Create a word cloud for posts with negative sentiment
negative_text = ' '.join(df[df['post_sentiment'] < 0]['selftext'].dropna())
negative_wordcloud = WordCloud(background_color='white').generate(negative_text)

plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Posts')
plt.show()

# Save the cleaned and analyzed data to a new CSV
df.to_csv('cleaned_reddit_data.csv', index=False)
