import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from vlm2edge.types import YoloBox

SYSTEM_PROMPT: str = (
    "You are an object detection annotator. Detect all visible trash/litter objects in the image. "
    "Return ONLY a valid JSON array of objects with format: "
    '[{"label": "litter", "bbox_2d": [xmin, ymin, xmax, ymax]}] '
    "where coordinates are integers normalized in [0, 1000]."
)


class QwenVLClient:
    """OpenAI-compatible client for Qwen3-VL served via vLLM or llama.cpp."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 60.0) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.client: httpx.Client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def wait_until_ready(self, timeout_seconds: int = 180, poll_interval: float = 3.0) -> None:
        """Block until the VLM server responds on /models, avoiding premature queue consumption."""
        deadline: float = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                resp: httpx.Response = self.client.get("/models", timeout=5.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(poll_interval)
        raise RuntimeError(f"VLM server at {self.base_url} did not become ready within {timeout_seconds}s")

    @staticmethod
    def _encode_image(image_path: Path, max_edge: int = 896) -> str:
        with Image.open(image_path) as img:
            transposed: Image.Image = ImageOps.exif_transpose(img)
            rgb_img: Image.Image = transposed.convert("RGB")
            rgb_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buffer: io.BytesIO = io.BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=85)
            encoded: str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def detect_boxes(self, image_path: Path, class_id: int = 0) -> list[YoloBox]:
        data_uri: str = self._encode_image(image_path)
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "Detect all litter items and return JSON bbox_2d."},
                    ],
                },
            ],
        }
        response: httpx.Response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        content: str = response.json()["choices"][0]["message"]["content"]
        return self.parse_qwen_boxes(content, class_id=class_id)

    @staticmethod
    def parse_qwen_boxes(raw_text: str, class_id: int = 0) -> list[YoloBox]:
        """Parse Qwen3-VL [0, 1000] bbox_2d JSON output into normalized YoloBox objects."""
        coord_list: list[list[float]] = []
        match: re.Match[str] | None = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
        if match is not None:
            try:
                items: list[dict[str, Any]] = json.loads(match.group(0))
                for item in items:
                    coords: list[float] | None = item.get("bbox_2d")
                    if coords and len(coords) == 4:
                        coord_list.append([float(v) for v in coords])
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        if not coord_list:
            pattern: str = (
                r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)'
                r"\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
            )
            for m in re.finditer(pattern, raw_text):
                coord_list.append([float(m.group(i)) for i in range(1, 5)])

        boxes: list[YoloBox] = []
        for coords in coord_list:
            x1, y1, x2, y2 = [max(0.0, min(1000.0, v)) / 1000.0 for v in coords]
            width: float = x2 - x1
            height: float = y2 - y1
            if width <= 0.002 or height <= 0.002:
                continue
            boxes.append(
                YoloBox(
                    class_id=class_id,
                    x_center=x1 + width / 2.0,
                    y_center=y1 + height / 2.0,
                    width=width,
                    height=height,
                )
            )
        return boxes
