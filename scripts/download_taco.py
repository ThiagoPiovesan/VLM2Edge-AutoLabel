import argparse
from pathlib import Path

from vlm2edge.datasets.taco import download_taco_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TACO dataset subset in YOLO format.")
    parser.add_argument("--dir", type=Path, default=Path("./data/taco"))
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()
    download_taco_subset(args.dir, limit=args.limit)
    print(f"TACO subset ready at {args.dir}")


if __name__ == "__main__":
    main()
