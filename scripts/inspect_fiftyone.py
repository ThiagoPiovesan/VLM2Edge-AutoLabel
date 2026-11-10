import os
from pathlib import Path

import fiftyone as fo  # type: ignore[import-untyped]

from vlm2edge.eval.metrics import match_boxes_greedy
from vlm2edge.types import YoloBox


def load_yolo_file(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts: list[str] = line.strip().split()
        if len(parts) != 5:
            continue
        boxes.append(
            YoloBox(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def to_fo_detections(boxes: list[YoloBox], label: str = "litter") -> fo.Detections:
    detections: list[fo.Detection] = []
    for box in boxes:
        x1, y1, x2, y2 = box.to_xyxy()
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=[x1, y1, x2 - x1, y2 - y1],
            )
        )
    return fo.Detections(detections=detections)


def main() -> None:
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "./data/taco")).resolve()
    images_dir: Path = dataset_dir / "images"
    gt_dir: Path = dataset_dir / "labels_gt"
    vlm_dir: Path = dataset_dir / "labels_vlm"

    dataset_name: str = "taco-vlm2edge-eval"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset: fo.Dataset = fo.Dataset(dataset_name)
    samples: list[fo.Sample] = []

    for img_path in sorted(images_dir.glob("*.jpg")):
        stem: str = img_path.stem
        gt_boxes: list[YoloBox] = load_yolo_file(gt_dir / f"{stem}.txt")
        vlm_boxes: list[YoloBox] = load_yolo_file(vlm_dir / f"{stem}.txt")
        tp, fp, fn = match_boxes_greedy(vlm_boxes, gt_boxes, iou_threshold=0.5)

        sample: fo.Sample = fo.Sample(filepath=str(img_path))
        sample["ground_truth"] = to_fo_detections(gt_boxes, label="litter_gt")
        sample["vlm_pred"] = to_fo_detections(vlm_boxes, label="litter_vlm")
        sample["tp"] = tp
        sample["fp"] = fp
        sample["fn"] = fn
        samples.append(sample)

    dataset.add_samples(samples)
    print(f"Loaded {len(samples)} samples into FiftyOne dataset '{dataset_name}'.")
    print("Opening FiftyOne App at http://localhost:5151 (Ctrl+C to exit)...")
    session: fo.Session = fo.launch_app(dataset, remote=True, address="0.0.0.0", port=5151)
    session.wait(-1)


if __name__ == "__main__":
    main()
