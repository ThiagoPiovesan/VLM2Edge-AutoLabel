import random
import shutil
from pathlib import Path

import yaml


def prepare_yolo_comparison_splits(
    dataset_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Create two YOLO dataset directories (`gt_train` and `vlm_train`) sharing the exact
    same image split and the exact same human Ground Truth validation labels (`labels_gt`).
    """
    images_dir: Path = dataset_dir / "images"
    gt_dir: Path = dataset_dir / "labels_gt"
    vlm_dir: Path = dataset_dir / "labels_vlm"

    valid_images: list[Path] = [
        img_path
        for img_path in sorted(images_dir.glob("*.jpg"))
        if (gt_dir / f"{img_path.stem}.txt").exists()
        and (vlm_dir / f"{img_path.stem}.txt").exists()
    ]
    if not valid_images:
        raise FileNotFoundError(
            f"No paired images with both labels_gt and labels_vlm found in {dataset_dir}"
        )

    rng: random.Random = random.Random(seed)
    shuffled: list[Path] = valid_images.copy()
    rng.shuffle(shuffled)

    val_count: int = max(1, int(len(shuffled) * val_ratio))
    val_images: list[Path] = sorted(shuffled[:val_count])
    train_images: list[Path] = sorted(shuffled[val_count:])

    splits_root: Path = dataset_dir / "yolo_splits"
    if splits_root.exists():
        shutil.rmtree(splits_root)

    yaml_paths: dict[str, Path] = {}
    for variant, train_label_src in (("gt", gt_dir), ("vlm", vlm_dir)):
        variant_dir: Path = splits_root / f"taco_{variant}"
        for subset, img_list in (("train", train_images), ("val", val_images)):
            img_out_dir: Path = variant_dir / "images" / subset
            lbl_out_dir: Path = variant_dir / "labels" / subset
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            # Validation ALWAYS uses human Ground Truth labels (labels_gt)
            label_source_dir: Path = train_label_src if subset == "train" else gt_dir

            for img_path in img_list:
                shutil.copy2(img_path, img_out_dir / img_path.name)
                src_lbl: Path = label_source_dir / f"{img_path.stem}.txt"
                shutil.copy2(src_lbl, lbl_out_dir / src_lbl.name)

        yaml_path: Path = variant_dir / "dataset.yaml"
        config: dict[str, object] = {
            "path": str(variant_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "litter"},
        }
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        yaml_paths[variant] = yaml_path

    return yaml_paths
