import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud

# Load the data
df = pd.read_csv('reddit_google_llm_gemini.csv',nrows=2000)

# Clean the data
df.drop_duplicates(inplace=True)
df['comment'] = df['comment'].fillna('N/A')
df.dropna(subset=['title', 'selftext'], inplace=True)
df['created'] = pd.to_datetime(df['created'], unit='s')

# NEW HUGGING FACE SENTIMENT PIPELINE 
print("Loading Hugging Face Transformer model (this may take a few seconds)...")
sentiment_pipeline = pipeline("sentiment-analysis", truncation=True, max_length=512)

def get_transformer_sentiment(text):
    # Handle empty or invalid text safely
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
        
    try:
        # Get result from the Hugging Face model
        result = sentiment_pipeline(text)[0]
        score = result['score']
        
        # Convert label to a polarity scale (-1.0 to 1.0)
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

# Visualizations

# 1. Post Sentiment Histogram
df['post_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Posts (Transformer Model)')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# 2. Comment Sentiment Histogram
df['comment_sentiment'].hist(bins=20)
plt.title('Sentiment Distribution for Comments (Transformer Model)')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# 3. Average Post Sentiment by Subreddit
subreddit_post_sentiment = df.groupby('url')['post_sentiment'].mean()
subreddit_post_sentiment.plot(kind='bar', title='Average Post Sentiment by Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45, ha='right') # Added slight rotation for better readability
plt.tight_layout()
plt.show()

# 4. Sentiment Over Time
df['created_date'] = pd.to_datetime(df['created']).dt.date
sentiment_over_time = df.groupby('created_date')['post_sentiment'].mean()

sentiment_over_time.plot(kind='line', title='Average Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.show()

# 5. Positive Word Cloud
print("Generating Positive Word Cloud...")
positive_text = ' '.join(df[df['post_sentiment'] > 0]['selftext'].dropna())
if positive_text.strip():
    positive_wordcloud = WordCloud(background_color='white', width=800, height=400).generate(positive_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Posts')
    plt.show()
else:
    print("Not enough positive text to generate a word cloud.")

# 6. Negative Word Cloud
print("Generating Negative Word Cloud...")
negative_text = ' '.join(df[df['post_sentiment'] < 0]['selftext'].dropna())
if negative_text.strip():
    negative_wordcloud = WordCloud(background_color='white', width=800, height=400).generate(negative_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Negative Posts')
    plt.show()
else:
    print("Not enough negative text to generate a word cloud.")

# Save cleaned and analyzed data
df.to_csv('cleaned_reddit_data_full.csv', index=False)
print("\nFull pipeline complete! Data saved to cleaned_reddit_data_full.csv")