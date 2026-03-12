import os
import time
from crewai import Agent, Task, Crew
from news_tool import fetch_news

# -------------------------------
# Set Groq API Key
# -------------------------------

os.environ["GROQ_API_KEY"] = "YGOUR_API_KEY"


# -------------------------------
# AGENT 1 : Research Agent
# -------------------------------

research_agent = Agent(
    role="News Researcher",
    goal="Understand the news articles and identify the main topic of each article",
    backstory=(
        "You are an expert news researcher who reads headlines and descriptions "
        "and understands the topic and context of each news article."
    ),
    llm="groq/llama-3.1-8b-instant",
    verbose=True
)


# -------------------------------
# AGENT 2 : Evaluation Agent
# -------------------------------

evaluation_agent = Agent(
    role="News Importance Analyst",
    goal="Evaluate the importance of news articles",
    backstory=(
        "You are a professional analyst who determines which news stories are important "
        "based on importance, impact and novelty."
    ),
    llm="groq/llama-3.1-8b-instant",
    verbose=True
)


# -------------------------------
# AGENT 3 : Summarizer Agent
# -------------------------------

summarizer_agent = Agent(
    role="News Summarizer",
    goal="Generate clear summaries for important news articles",
    backstory=(
        "You are an expert journalist who writes concise and clear summaries "
        "for important news stories."
    ),
    llm="groq/llama-3.1-8b-instant",
    max_iter=1,
    verbose=True
)


# -------------------------------
# Fetch News (Tool)
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
# TASK 1 : Research Task
# -------------------------------

research_task = Task(
    description=f"""
Read the following news articles and understand their topics.

{news_text}

For each article identify the main topic and category.

Return format:

Article #
Topic
Category
""",
    expected_output="List of articles with topics and categories",
    agent=research_agent
)


# -------------------------------
# TASK 2 : Evaluation Task
# -------------------------------

evaluation_task = Task(
    description="""
Evaluate the articles provided by the researcher.

Score each article based on:

Importance (0-3)
Impact (0-3)
Novelty (0-2)

Total Score = sum

Rules:
Score >= 4 → important
Score < 4 → skip

Return format:

Article #
Score
Decision (IMPORTANT / SKIP)
""",
    expected_output="Articles scored by importance",
    agent=evaluation_agent,
    context=[research_task]
)


# -------------------------------
# TASK 3 : Summarization Task
# -------------------------------

summary_task = Task(
    description="""
You are given:

1. The original news articles from the researcher
2. The evaluation results from the analyst

Your job:

• Identify ONLY articles marked IMPORTANT
• Ignore articles marked SKIP
• Using the article title and description, write a 2–3 sentence summary

Return format:

Article #
Summary

Example:

Article 1
Summary: Oil prices dropped as markets reacted to geopolitical tensions in the Strait of Hormuz, despite warnings of larger U.S. strikes on Iran. Analysts say continued instability could impact global energy supply and pricing.

Article 2
Summary: The United States launched one of its most intense rounds of strikes against Iranian targets while Iran reduced missile activity. The development signals escalating tensions and potential regional instability.
""",
    expected_output="Summaries of important articles",
    agent=summarizer_agent,
    context=[research_task, evaluation_task]
)


# -------------------------------
# CREATE CREW (THIS WAS MISSING)
# -------------------------------

crew = Crew(
    agents=[research_agent, evaluation_agent, summarizer_agent],
    tasks=[research_task, evaluation_task, summary_task],
    verbose=True
)


# -------------------------------
# Rate-limit safe execution
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