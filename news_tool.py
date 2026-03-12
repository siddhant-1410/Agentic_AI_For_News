import requests

API_KEY = "NEWS_API_KEY"  # Replace with your NewsAPI key

def fetch_news():

    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"

    response = requests.get(url).json()

    articles = response["articles"][:5]

    news = []

    for article in articles:
        news.append({
            "title": article["title"],
            "description": article["description"]
        })

    return news