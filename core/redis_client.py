import os
import logging
import redis
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def _create_redis_client():
    try:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Could not connect to Redis at {REDIS_URL}: {e}")
        return None

redis_client = _create_redis_client()
