import requests
import urllib.parse
import pandas as pd
import os, json
import time
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv


def retry_on_openai_timeout(max_retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except openai.error.Timeout as e:
                    retries += 1
                    print(f"Timeout error occurred. Retrying {retries}/{max_retries} in {delay} seconds...")
                    time.sleep(delay)
                    if retries == max_retries:
                        print("Max retries reached. Raising error.")
                        raise e

        return wrapper

    return decorator


class CryptoNewsAnalyzer:
    def __init__(self, tokens, api_key, start_date, end_date):
        """
        Initialize the class with a list of tokens and the NewsAPI key.

        Parameters:
            tokens (list): List of cryptocurrency tokens (e.g., ["AAVE", "LINK", "BTC"]).
            api_key (str): Your NewsAPI key.
        """
        self.tokens = tokens
        self.api_key = api_key
        self.domains = (
            "coindesk.com,cointelegraph.com,decrypt.co,"
            "cryptoslate.com,bitcoinmagazine.com,newsbtc.com,"
            "cryptobriefing.com,theblock.co,ambcrypto.com"
        )
        self.news_data = {token: [] for token in tokens}

        # Extended token names mapping
        self.extended_names = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum"
        }

        # Set dates
        self.start_date = start_date
        self.end_date = end_date

    def get_news(self, token):
        """
        Fetch news articles related to a specific token.

        Parameters:
            token (str): The cryptocurrency token to fetch news for.
        """
        # Get extended name if exists, otherwise default to token name
        extended_name = self.extended_names.get(token.upper(), token)

        # Define keywords for the token, excluding extended name if it matches the token
        keywords = f"{token} OR {token.lower()}"
        if extended_name.lower() != token.lower():  # Only include extended name if different
            keywords += f" OR {extended_name} OR {extended_name.lower()}"
        keywords += "OR crypto OR cryptocurrency OR blockchain"
        encoded_keywords = urllib.parse.quote(keywords.strip())

        # # Define date range
        # self.end_date = datetime.now()  # Current date as end date
        # self.start_date = self.end_date - timedelta(days=30)  # Start date, 30 days back from today

        # Format dates as 'YYYY-MM-DD'
        formatted_start_date = self.start_date.strftime('%Y-%m-%d')
        formatted_end_date = self.end_date.strftime('%Y-%m-%d')

        # Define the CSV file name
        file_name = f"{self.start_date.strftime('%Y-%m-%d')}_to_{self.end_date.strftime('%Y-%m-%d')}_everything_{token}.csv"
        token_dir = f"./{token}"
        file_path = os.path.join(token_dir, file_name)

        # Check if the CSV file already exists
        if os.path.exists(file_path):
            print(f"Using existing data from {file_path}")
            df = pd.read_csv(file_path)
            self.news_data[token] = [{"news": row.to_dict(), "top_headlines": False} for _, row in df.iterrows()]
            return

        # Fetching everything news for the token
        everything_response = requests.get(
            f"https://newsapi.org/v2/everything?q={encoded_keywords}&from={formatted_start_date}&to={formatted_end_date}"
            f"&domains={self.domains}&language=en&apiKey={self.api_key}"
        )
        everything_articles = everything_response.json().get("articles", [])
        self.save_to_csv(everything_articles, token, "everything")

        # Store articles in news_data
        for article in everything_articles:
            article_info = {
                "news": article,
                "top_headlines": False  # Top headlines are not being used
            }
            self.news_data[token].append(article_info)

    def save_to_csv(self, articles, token, request_type):
        """
        Save articles to CSV file in a dedicated folder for each token.

        Parameters:
            articles (list): List of articles to save.
            token (str): The token related to the articles.
            request_type (str): Type of request (everything).
        """
        # Create a directory for the token if it doesn't exist
        token_dir = f"./{token}"
        os.makedirs(token_dir, exist_ok=True)

        if articles:
            df = pd.DataFrame(articles)
            period = f"{self.start_date.strftime('%Y-%m-%d')}_to_{self.end_date.strftime('%Y-%m-%d')}_{request_type}_{token}.csv"
            file_path = os.path.join(token_dir, period)
            df.to_csv(file_path, index=False)

    def analyze_news_sentiment(self):
        """
        Analyze news articles for each token and determine their sentiment and historical context.
        """
        for token, articles in self.news_data.items():
            # Define the CSV file name for sentiment data
            file_name = f"{self.start_date.strftime('%Y-%m-%d')}_to_{self.end_date.strftime('%Y-%m-%d')}_sentiment_{token}.csv"
            token_dir = f"./{token}"
            file_path = os.path.join(token_dir, file_name)

            # Check if the sentiment CSV file already exists
            if os.path.exists(file_path):
                print(f"Loading existing sentiment data from {file_path}")
                # Load existing data and update articles with it
                existing_data = pd.read_csv(file_path)
                for article, (_, row) in zip(articles, existing_data.iterrows()):
                    article['is_historical'] = row['is_historical']
                    article['sentiment'] = row['sentiment']
                continue  # Skip further analysis since data was loaded

            for article in articles:
                title = article["news"].get("title", "")
                description = article["news"].get("description", "")
                content = article["news"].get("content", "")

                # Combine title, description, and content for sentiment analysis
                news_text = f"{title} {description} {content}"

                # Evaluate sentiment and historical context
                is_historical, sentiment = self.evaluate_news_sentiment(news_text, token)

                # Add results to the article info
                article['is_historical'] = is_historical
                article['sentiment'] = sentiment

    # Applying the decorator
    @retry_on_openai_timeout()
    def evaluate_news_sentiment(self, news_text, token):
        prompt = f"""
        Analyze the following news article for its potential impact on {token} (e.g., significant rise, rise, neutral, fall, significant fall) 
        and determine if it’s mainly a factual report (i.e., about past price movements) or suggests future implications.

        News Article:
        {news_text}

        Based on the context, answer the following questions:
        1. Is this article mainly a factual report about past events? Respond with "True" or "False".
        2. What sentiment does this news imply for the future price movement of {token}? Choose one: "significant rise", "rise", "neutral", "fall", or "significant fall".
        """

        # Send the prompt to the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content.strip().lower()

        is_historical = "true" in response_text
        sentiment = None
        for option in ["significant rise", "rise", "neutral", "fall", "significant fall"]:
            if option in response_text:
                sentiment = option
                break

        return is_historical, sentiment

    def save_sentiment_to_csv(self):
        """
        Save analyzed news data with sentiment and historical context to a new CSV file.
        """

        for token, articles in self.news_data.items():
            if not articles:
                continue

            # Define the new CSV file name
            file_name = f"{self.start_date.strftime('%Y-%m-%d')}_to_{self.end_date.strftime('%Y-%m-%d')}_sentiment_{token}.csv"
            token_dir = f"./{token}"
            file_path = os.path.join(token_dir, file_name)

            # Check if the sentiment CSV file already exists
            if os.path.exists(file_path):
                print(f"Using existing sentiment data from {file_path}")
                continue  # Skip save_sentiment_to_csv for current token

            # Prepare data for saving
            results = []
            for article in articles:
                results.append({
                    'source': article["news"]["source"],
                    'author': article["news"]["author"],
                    'title': article["news"]["title"],
                    'description': article["news"]["description"],
                    'publishedAt': article["news"]["publishedAt"],
                    'url': article["news"]["url"],
                    'content': article["news"]["content"],
                    'is_historical': article['is_historical'],
                    'sentiment': article['sentiment']
                })

            # Convert results to DataFrame
            df = pd.DataFrame(results)

            # Save to CSV
            df.to_csv(file_path, index=False)

    def analyze_news(self):
        """
        Analyze the collected news for each token, separated into significant rise and fall events,
        and sorted by date. Also, save the rise and fall dates in JSON files if they don't already exist.
        """
        for token, articles in self.news_data.items():
            # Separate and sort news into 'rise' and 'fall' based on sentiment and date
            rise_articles = sorted(
                [article for article in articles
                 if article.get("is_historical") == False and article["sentiment"] == 'significant rise'],
                key=lambda x: x["news"]["publishedAt"]
            )
            fall_articles = sorted(
                [article for article in articles
                 if article.get("is_historical") == False and article["sentiment"] == 'fall'],
                key=lambda x: x["news"]["publishedAt"]
            )

            count_rise_news = len(rise_articles)
            count_fall_news = len(fall_articles)
            rise_dates = [article["news"]["publishedAt"] for article in rise_articles]
            fall_dates = [article["news"]["publishedAt"] for article in fall_articles]

            print(f"Token: {token}")
            print(f"Total news: {len(articles)}")

            # Print rise news information if any
            if count_rise_news > 0:
                print(f"Total rise impact news: {count_rise_news}")
                print(f"Rise impact news period: {min(rise_dates)} to {max(rise_dates) if rise_dates else 'N/A'}")
                print(f"Rise impact news dates (sorted): {rise_dates}\n")
            else:
                print("No significant rise impact news.\n")

            # Print fall news information if any
            if count_fall_news > 0:
                print(f"Total fall impact news: {count_fall_news}")
                print(f"Fall impact news period: {min(fall_dates)} to {max(fall_dates) if fall_dates else 'N/A'}")
                print(f"Fall impact news dates (sorted): {fall_dates}\n")
            else:
                print("No significant fall impact news.\n")

            # Define the JSON file names
            file_name_rise = f"{self.start_date.strftime('%Y-%m-%d')}_to_{self.end_date.strftime('%Y-%m-%d')}_rise_dates_{token}.json"
            file_name_fall = f"{self.start_date.strftime('%Y-%m-%d')}_to_{self.end_date.strftime('%Y-%m-%d')}_fall_dates_{token}.json"
            token_dir = f"./{token}"

            # Ensure the token directory exists
            os.makedirs(token_dir, exist_ok=True)

            # Define the file paths
            file_path_rise = os.path.join(token_dir, file_name_rise)
            file_path_fall = os.path.join(token_dir, file_name_fall)

            # Save rise_dates to JSON if file does not exist
            if not os.path.exists(file_path_rise):
                with open(file_path_rise, 'w') as file:
                    json.dump(rise_dates, file, indent=4)
                print(f"Rise dates saved to {file_path_rise}")
            else:
                print(f"Rise dates file already exists: {file_path_rise}, skipping save.")

            # Save fall_dates to JSON if file does not exist
            if not os.path.exists(file_path_fall):
                with open(file_path_fall, 'w') as file:
                    json.dump(fall_dates, file, indent=4)
                print(f"Fall dates saved to {file_path_fall}")
            else:
                print(f"Fall dates file already exists: {file_path_fall}, skipping save.")


if __name__ == "__main__":
    # Load environment variables from api.env
    # OPENAI_API_KEY=sk-proj-NvmRe*****************
    # NEWS_API_KEY=b5*******************
    load_dotenv("api_key.env")

    # Instantiate the OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    api_key = os.getenv("NEWS_API_KEY")

    # Define the tokens, start date, and end date for news analysis
    tokens = ["BTC", "ETH", "LINK"]

    # Set the backtesting period
    start_date = datetime(2024, 9, 27)
    end_date = datetime(2024, 10, 6)

    # Initialize the news analyzer with tokens, API key, and date range
    news_analyzer = CryptoNewsAnalyzer(tokens, api_key, start_date, end_date)

    # Fetch news for each token within the specified date range
    for token in tokens:
        news_analyzer.get_news(token)

    # Analyze the sentiment of the collected news articles
    news_analyzer.analyze_news_sentiment()

    # Save the sentiment analysis results to CSV for future reference
    news_analyzer.save_sentiment_to_csv()

    # Further analyze the news data to determine significant rise/fall events
    news_analyzer.analyze_news()
