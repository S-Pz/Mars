import praw
import json
import asyncio
import asyncpraw

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

CLIENT_ID = 'NdcGRdebTz5sHJGIAScpoA'
CLIENT_SECRET = 'GQeAt0Wu5bHVZEc4DFwMe9a7b6Le_g'

user_agent = "Scraper 1.0 by /u/Funny-Two2"
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=user_agent
)

def fetch_recent_posts_from_reddit(time_limit_hours=3):
    posts_data = []
    
    print("Coletando posts de todo o Reddit")
    subreddit = reddit.subreddit("all")  # Aguarda o objeto subreddit

    for submission in subreddit.hot(limit=None):  # Pega posts mais recentes de todo o Reddit
        submission.comments.replace_more(limit=32)
        for top_level_comment in submission.comments:
            print(submission.title)
            print(top_level_comment.body)
            print(top_level_comment.author)

            post_info = {
                "title": submission.title,
                "subredidt" : str(submission.subreddit),
                "comments" : top_level_comment.body,
                "comments_auth": str(top_level_comment.author)
            }
            posts_data.append(post_info)

        # Adicione um intervalo para evitar a limitação de taxa
        # Salve todos os posts em um único arquivo JSON
        with open(f'posts_data.json', 'a', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=4)
        
        posts_data = []
        # Aguarda um intervalo para evitar a limitação de taxa

    print(f"Número total de posts coletados: {len(posts_data)}")

# Execução no ambiente Jupyter
if __name__ == "__main__":
    
    try:
        fetch_recent_posts_from_reddit(time_limit_hours=3)
    except RuntimeError as e:
        print(f"Erro: {e}")