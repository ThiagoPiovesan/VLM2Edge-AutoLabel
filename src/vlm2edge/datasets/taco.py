import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from vlm2edge.types import YoloBox

TACO_ANNOTATIONS_URL: str = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)


def download_taco_subset(target_dir: Path, limit: int = 150) -> None:
    """Download TACO COCO annotations and a subset of images + convert GT to single-class YOLO."""
    images_dir: Path = target_dir / "images"
    gt_labels_dir: Path = target_dir / "labels_gt"
    vlm_labels_dir: Path = target_dir / "labels_vlm"
    for folder in (images_dir, gt_labels_dir, vlm_labels_dir):
        folder.mkdir(parents=True, exist_ok=True)

    ann_path: Path = target_dir / "annotations.json"
    if not ann_path.exists():
        print(f"Downloading TACO annotations -> {ann_path}")
        resp: httpx.Response = httpx.get(TACO_ANNOTATIONS_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        ann_path.write_text(resp.text, encoding="utf-8")

    coco: dict[str, Any] = json.loads(ann_path.read_text(encoding="utf-8"))
    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[int(ann["image_id"])].append(ann)

    images: list[dict[str, Any]] = coco["images"][:limit]
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for img_info in tqdm(images, desc="Downloading TACO images"):
            img_id: int = int(img_info["id"])
            img_w: float = float(img_info["width"])
            img_h: float = float(img_info["height"])
            url: str = img_info.get("flickr_url") or img_info.get("flickr_640_url") or ""
            if not url:
                continue

            stem: str = f"taco_{img_id:05d}"
            img_path: Path = images_dir / f"{stem}.jpg"
            gt_path: Path = gt_labels_dir / f"{stem}.txt"

            if not img_path.exists():
                try:
                    img_resp: httpx.Response = client.get(url)
                    img_resp.raise_for_status()
                    img_path.write_bytes(img_resp.content)
                except (httpx.HTTPError, OSError) as exc:
                    print(f"Skipping image {img_id}: {exc}")
                    continue

            lines: list[str] = []
            for ann in anns_by_img[img_id]:
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 1.0 or h <= 1.0:
                    continue
                box = YoloBox(
                    class_id=0,
                    x_center=max(0.0, min(1.0, (x + w / 2.0) / img_w)),
                    y_center=max(0.0, min(1.0, (y + h / 2.0) / img_h)),
                    width=max(0.001, min(1.0, w / img_w)),
                    height=max(0.001, min(1.0, h / img_h)),
                )
                lines.append(box.to_yolo_line())
            gt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
