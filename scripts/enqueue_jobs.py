import os
from pathlib import Path

from vlm2edge.queue.redis_queue import AutoLabelQueue
from vlm2edge.types import LabelJob


def main() -> None:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    model_name: str = os.getenv("VLM_MODEL", "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit")
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "./data/taco"))
    images_dir: Path = dataset_dir / "images"
    labels_vlm_dir: Path = dataset_dir / "labels_vlm"
    labels_vlm_dir.mkdir(parents=True, exist_ok=True)

    queue = AutoLabelQueue(redis_url=redis_url)
    enqueued_count: int = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        job_id: str = queue.build_job_id(str(img_path), model_name)
        out_label: Path = labels_vlm_dir / f"{img_path.stem}.txt"
        job = LabelJob(
            job_id=job_id,
            image_path=str(img_path),
            output_label_path=str(out_label),
        )
        if queue.enqueue(job):
            enqueued_count += 1

    print(f"Enqueued {enqueued_count} jobs. Queue stats: {queue.stats()}")


if __name__ == "__main__":
    main()
