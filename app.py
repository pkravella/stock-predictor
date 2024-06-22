from flask import Flask, request, render_template
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import praw
import nltk
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)

# Setup Reddit API client
#(Code hidden for privacy reasons)
 
# Preprocess text function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Analyze sentiment function
analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

# Scrape posts function
def scrape_posts(subreddit, stock_symbol, start_time, end_time):
    subreddit = reddit.subreddit(subreddit)
    posts = []
    for post in subreddit.search(f"{stock_symbol} OR ${stock_symbol}", sort='new'):
        post_time = datetime.fromtimestamp(post.created_utc, pytz.UTC)
        if start_time <= post_time <= end_time:
            posts.append([post.id, post.title, post.selftext, post_time])
    return pd.DataFrame(posts, columns=['id', 'title', 'body', 'created'])

# Prediction function
def predict_stock_price(stock_symbol):
    timezone = pytz.UTC
    end_time = datetime.now(timezone)
    start_time = end_time - timedelta(days=30)

    df = scrape_posts('wallstreetbets', stock_symbol, start_time, end_time)
    if df.empty:
        print("No posts found for this stock symbol.")
        return None, "No posts found for this stock symbol."

    df['cleaned_title'] = df['title'].apply(preprocess_text)
    df['cleaned_body'] = df['body'].apply(preprocess_text)
    df['title_sentiment'] = df['cleaned_title'].apply(analyze_sentiment)
    df['body_sentiment'] = df['cleaned_body'].apply(analyze_sentiment)
    df['overall_sentiment'] = df[['title_sentiment', 'body_sentiment']].mean(axis=1)
    df['date'] = pd.to_datetime(df['created']).dt.date
    daily_sentiment = df.groupby('date')['overall_sentiment'].mean().reset_index()

    print("Daily Sentiment DataFrame:")
    print(daily_sentiment)

    stock_data = yf.download(stock_symbol, start=start_time.strftime('%Y-%m-%d'), end=end_time.strftime('%Y-%m-%d'))
    if stock_data.empty:
        print("No matching stock data found.")
        return None, "No matching stock data found."
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = stock_data['Date'].dt.date

    print("Stock Data DataFrame:")
    print(stock_data)

    data = pd.merge(daily_sentiment, stock_data[['Date', 'Close', 'Volume']], left_on='date', right_on='Date')
    
    print("Merged Data DataFrame:")
    print(data)

    if data.empty:
        print("No matching merged data found.")
        return None, "No matching merged data found."
    
    X = data[['overall_sentiment', 'Volume']]
    y = data['Close']
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Not enough data to split.")
        return None, "Not enough data to split."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return y_pred[-1], mse

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    prediction, mse = predict_stock_price(stock_symbol)
    if prediction is None:
        return render_template('index.html', error=mse)
    return render_template('index.html', stock_symbol=stock_symbol, prediction=prediction, mse=mse)

if __name__ == '__main__':
    app.run(debug=True)
