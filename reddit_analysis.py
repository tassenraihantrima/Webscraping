import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the data
df = pd.read_csv('reddit_google_llm_gemini.csv')

# Clean the data
df.drop_duplicates(inplace=True)
df['comment'] = df['comment'].fillna('N/A')
df.dropna(subset=['title', 'selftext'], inplace=True)
df['created'] = pd.to_datetime(df['created'], unit='s')

# Print basic stats
top_posts = df.sort_values(by='score', ascending=False).head(10)
print("Top 10 posts:")
print(top_posts[['title', 'score', 'url']])

subreddit_counts = df['url'].groupby(df['url']).count()
print("\nPost count by subreddit:")
print(subreddit_counts)

# NEW HUGGING FACE SENTIMENT PIPELINE 
print("\nLoading Hugging Face Transformer model (this may take a few seconds)...")
sentiment_pipeline = pipeline("sentiment-analysis", truncation=True, max_length=512)

def get_transformer_sentiment(text):
    # Handle empty or invalid text safely
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
        
    try:
        # Get result from the Hugging Face model
        result = sentiment_pipeline(text)[0]
        score = result['score']
        
        # Convert label to a polarity scale (-1.0 to 1.0) to match your previous setup
        if result['label'] == 'POSITIVE':
            return score
        elif result['label'] == 'NEGATIVE':
            return -score
        else:
            return 0.0
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0.0

print("Analyzing post sentiment...")
df['post_sentiment'] = df['selftext'].apply(get_transformer_sentiment)

print("Analyzing comment sentiment...")
df['comment_sentiment'] = df['comment'].apply(get_transformer_sentiment)

print("\nAverage post sentiment:", df['post_sentiment'].mean())
print("Average comment sentiment:", df['comment_sentiment'].mean())

# Visualizations
# 1. Sentiment Histogram
df['post_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Posts (Transformer Model)')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# 2. Posts Over Time
df['created_date'] = pd.to_datetime(df['created']).dt.date
post_counts = df.groupby('created_date').size()

post_counts.plot(kind='line', title='Number of Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.show()

# 3. Word Cloud
all_text = ' '.join(df['selftext'].dropna())
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Save cleaned and analyzed data
df.to_csv('cleaned_reddit_data.csv', index=False)
print("\nPipeline complete! Saved to cleaned_reddit_data.csv")