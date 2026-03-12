import os
import time
from crewai import Agent, Task, Crew
from news_tool import fetch_news

# -------------------------------
# Set Groq API key
# -------------------------------

os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"

# -------------------------------
# Create Agent
# -------------------------------

news_agent = Agent(
    role="News Analyst",
    goal="Identify important news and summarize them",
    backstory=(
        "You are a professional news analyst who evaluates news importance "
        "and summarizes only the important stories."
    ),
    llm="groq/openai/gpt-oss-safeguard-20b",
    max_iter=1,
    verbose=True
)

# -------------------------------
# Fetch News
# -------------------------------

news_list = fetch_news()

news_text = ""

for i, article in enumerate(news_list, start=1):
    news_text += f"""
Article {i}
Title: {article['title']}
Description: {article['description']}
"""

# -------------------------------
# Task
# -------------------------------

task = Task(
    description=f"""
Evaluate the following news articles:

{news_text}

For EACH article assign:

Importance (0-3)
Impact (0-3)
Novelty (0-2)

Total Score = sum.

Rules:
If score >= 4 → summarize
If score < 4 → skip.

Return results in this format:

Article #
Score:
Decision: SUMMARIZE / SKIP
Summary (if summarized)
""",
    expected_output="List of scored and summarized articles",
    agent=news_agent
)

# -------------------------------
# Crew
# -------------------------------

crew = Crew(
    agents=[news_agent],
    tasks=[task],
    verbose=True
)

# -------------------------------
# Retry logic for rate limits
# -------------------------------

max_retries = 5
delay = 5

for attempt in range(max_retries):
    try:
        result = crew.kickoff()
        print("\nFINAL RESULT\n")
        print(result)
        break

    except Exception as e:
        if "RateLimit" in str(e):
            print(f"\nRate limit hit. Waiting {delay} seconds...\n")
            time.sleep(delay)
            delay *= 2
        else:
            raise