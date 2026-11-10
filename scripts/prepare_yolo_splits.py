import argparse
import os
from pathlib import Path

from vlm2edge.datasets.splits import prepare_yolo_comparison_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare paired YOLO train/val splits for GT vs VLM auto-label comparison."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(os.getenv("DATASET_DIR", "./data/taco")),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    yaml_paths: dict[str, Path] = prepare_yolo_comparison_splits(
        dataset_dir=args.dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Prepared YOLO splits in {args.dataset_dir / 'yolo_splits'}:")
    for variant, path in yaml_paths.items():
        print(f"  [{variant.upper()}] {path}")


if __name__ == "__main__":
    main()
