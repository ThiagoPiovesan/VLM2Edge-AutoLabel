import hashlib

from redis import Redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from vlm2edge.types import LabelJob


class AutoLabelQueue:
    """Resilient Redis queue with idempotency tracking, retry limit, and Dead-Letter Queue (DLQ)."""

    PENDING_KEY: str = "vlm2edge:queue:pending"
    DLQ_KEY: str = "vlm2edge:queue:dlq"
    DONE_SET_KEY: str = "vlm2edge:done:set"

    def __init__(self, redis_url: str, max_retries: int = 3) -> None:
        self.client: Redis = Redis.from_url(redis_url, decode_responses=True)
        self.max_retries: int = max_retries

    @staticmethod
    def build_job_id(image_path: str, model_name: str) -> str:
        raw_key: str = f"{image_path}::{model_name}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def enqueue(self, job: LabelJob) -> bool:
        """Enqueue job if not already completed (idempotent)."""
        if self.client.sismember(self.DONE_SET_KEY, job.job_id):
            return False
        self.client.rpush(self.PENDING_KEY, job.model_dump_json())
        return True

    def pop_blocking(self, timeout_seconds: int = 5) -> LabelJob | None:
        try:
            result: tuple[str, str] | None = self.client.blpop(
                [self.PENDING_KEY], timeout=timeout_seconds
            )  # type: ignore[assignment]
        except RedisTimeoutError:
            return None
        if result is None:
            return None
        _, payload = result
        return LabelJob.model_validate_json(payload)

    def mark_done(self, job: LabelJob) -> None:
        self.client.sadd(self.DONE_SET_KEY, job.job_id)

    def handle_failure(self, job: LabelJob, error_message: str) -> str:
        """Retry job or move to Dead-Letter Queue when retries are exhausted."""
        next_attempt: int = job.attempt + 1
        updated_job: LabelJob = job.model_copy(
            update={"attempt": next_attempt, "last_error": error_message}
        )
        if next_attempt < self.max_retries:
            self.client.rpush(self.PENDING_KEY, updated_job.model_dump_json())
            return "retried"
        self.client.rpush(self.DLQ_KEY, updated_job.model_dump_json())
        return "dlq"

    def stats(self) -> dict[str, int]:
        return {
            "pending": int(self.client.llen(self.PENDING_KEY)),  # type: ignore[arg-type]
            "done": int(self.client.scard(self.DONE_SET_KEY)),  # type: ignore[arg-type]
            "dlq": int(self.client.llen(self.DLQ_KEY)),  # type: ignore[arg-type]
        }
