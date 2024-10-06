
import praw
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('reddit_google_llm_gemini.csv')

# Data Cleaning and Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing comment text with 'N/A'
df['comment'].fillna('N/A', inplace=True)

# Drop rows where title or selftext is missing
df.dropna(subset=['title', 'selftext'], inplace=True)

# Convert Unix timestamp to readable date
df['created'] = pd.to_datetime(df['created'], unit='s')

# Exploratory Data Analysis (EDA)
# Top 10 posts by score
top_posts = df.sort_values(by='score', ascending=False).head(10)
print("Top 10 posts:")
print(top_posts[['title', 'score', 'url']])

# Number of posts by subreddit
subreddit_counts = df['url'].groupby(df['url']).count()
print("Post count by subreddit:")
print(subreddit_counts)

# Sentiment Analysis
# Define a function to calculate sentiment using TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the function to posts and comments
df['post_sentiment'] = df['selftext'].apply(get_sentiment)
df['comment_sentiment'] = df['comment'].apply(get_sentiment)

# Average sentiment scores
print("Average post sentiment:", df['post_sentiment'].mean())
print("Average comment sentiment:", df['comment_sentiment'].mean())

# Visualize Sentiment Distribution
df['post_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Posts')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Visualize Post Frequency Over Time
# Group posts by creation date
df['created'] = pd.to_datetime(df['created'])
post_counts = df.groupby(df['created'].dt.date).size()

# Plot the time series
post_counts.plot(kind='line', title='Number of Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.show()

# Word Cloud for Most Common Words
# Generate a word cloud for the selftext (posts)
all_text = ' '.join(df['selftext'].dropna())
wordcloud = WordCloud(background_color='white').generate(all_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Save the cleaned data to a new CSV
df.to_csv('cleaned_reddit_data.csv', index=False)

