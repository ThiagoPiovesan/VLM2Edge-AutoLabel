import os
from pathlib import Path

import httpx

from vlm2edge.queue.redis_queue import AutoLabelQueue
from vlm2edge.types import LabelJob, YoloBox
from vlm2edge.vlm.client import QwenVLClient


def main() -> None:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    vlm_base_url: str = os.getenv("VLM_BASE_URL", "http://localhost:8000/v1")
    vlm_model: str = os.getenv("VLM_MODEL", "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit")

    queue = AutoLabelQueue(redis_url=redis_url, max_retries=3)
    vlm = QwenVLClient(base_url=vlm_base_url, model=vlm_model)

    print(f"Waiting for VLM server at {vlm_base_url}...")
    vlm.wait_until_ready()
    print(f"Worker started. Listening on {redis_url} -> VLM {vlm_base_url} ({vlm_model})")
    while True:
        job: LabelJob | None = queue.pop_blocking(timeout_seconds=5)
        if job is None:
            print(f"Queue idle. Stats: {queue.stats()}")
            break
        try:
            boxes: list[YoloBox] = vlm.detect_boxes(Path(job.image_path))
            out_path: Path = Path(job.output_label_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = [b.to_yolo_line() for b in boxes]
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            queue.mark_done(job)
            print(f"[OK] {job.job_id} ({ len(boxes) } boxes) -> {out_path.name}")
        except (httpx.HTTPError, ValueError, OSError, RuntimeError) as exc:
            status: str = queue.handle_failure(job, str(exc))
            print(f"[FAIL -> {status.upper()}] {job.job_id} (attempt {job.attempt + 1}): {exc}")


if __name__ == "__main__":
    main()
