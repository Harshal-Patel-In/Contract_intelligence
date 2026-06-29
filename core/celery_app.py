import os, sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # Load ST model from cache — skip HF Hub 30s network checks
os.environ["HF_DATASETS_OFFLINE"] = "1"     # Same — prevent huggingface_hub background polling
# Ensure the project root is in sys.path so all local packages are importable
# regardless of how the worker/server is launched (uv run, direct python, etc.)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ocr_worker",
    broker = REDIS_URL,
    backend = REDIS_URL,
    include = ["tasks.pipeline_tasks"]
)


celery_app.conf.update(
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=100,
    task_time_limit=3600,
    task_soft_time_limit=3300,
)

