"""Fetch posts from JSONPlaceholder API."""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from core.config import AppConfig
from core.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class Post(BaseModel):
    id: int
    userId: int
    title: str
    body: str

    def formatted_content(self) -> str:
        return f"Title: {self.title}\n\n{self.body}"


def fetch_posts(config: AppConfig) -> list[Post]:
    """Download the first N posts from JSONPlaceholder."""

    def _fetch() -> list[Post]:
        url = config.api.posts_url
        params = {"_limit": config.api.posts_limit}
        logger.info("Fetching posts from %s (limit=%d)", url, config.api.posts_limit)
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        posts = [Post.model_validate(item) for item in data]
        logger.info("Fetched %d posts", len(posts))
        return posts

    return retry_with_backoff(_fetch, max_attempts=3)
