#!/usr/bin/env python3
"""Rerun GroundingDINO and render its pudding detections without supervision."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from groundingdino.util.inference import load_image, load_model, predict


TASK_MARKERS = (
    "between_the_plate_and_the_ramekin",
    "on_the_ramekin",
    "on_the_stove",
    "on_the_wooden_cabinet",
)


def _font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _task_index(path):
    matches = [index for index, marker in enumerate(TASK_MARKERS) if marker in str(path)]
    if len(matches) != 1:
        raise RuntimeError("Cannot identify task from {}".format(path))
    return matches[0]


def _xyxy_pixels(box, width, height):
    cx, cy, box_width, box_height = [float(value) for value in box]
    x1 = max(0, min(width - 1, int(round((cx - box_width / 2.0) * width))))
    y1 = max(0, min(height - 1, int(round((cy - box_height / 2.0) * height))))
    x2 = max(0, min(width - 1, int(round((cx + box_width / 2.0) * width))))
    y2 = max(0, min(height - 1, int(round((cy + box_height / 2.0) * height))))
    return [x1, y1, x2, y2]


def _draw_detection(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    label_font = _font(24)
    colors = ((0, 255, 80), (255, 210, 0), (0, 200, 255), (255, 80, 180))
    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = detection["box_xyxy_pixels"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=6)
        label = "{} {:.3f}".format(detection["phrase"], detection["confidence"])
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_y = max(0, y1 - label_height - 12)
        draw.rectangle(
            (x1, label_y, min(annotated.width - 1, x1 + label_width + 14), y1),
            fill=color,
        )
        draw.text((x1 + 7, label_y + 3), label, fill=(0, 0, 0), font=label_font)
    return annotated


def _side_by_side(raw, annotated, task_index, view):
    header_height = 54
    canvas = Image.new("RGB", (raw.width * 2, raw.height + header_height), "white")
    canvas.paste(raw, (0, header_height))
    canvas.paste(annotated, (raw.width, header_height))
    draw = ImageDraw.Draw(canvas)
    header_font = _font(28)
    draw.text((18, 10), "Task {} · {} · raw view".format(task_index, view), fill="black", font=header_font)
    draw.text(
        (raw.width + 18, 10),
        "GroundingDINO · prompt: chocolate pudding",
        fill="black",
        font=header_font,
    )
    return canvas


def _contact_sheet(paths, destination):
    images = [Image.open(str(path)).convert("RGB") for path in paths]
    target_width = 1400
    resized = []
    for image in images:
        height = int(round(image.height * target_width / image.width))
        resized.append(image.resize((target_width, height), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (target_width, sum(image.height for image in resized)), "white")
    y = 0
    for image in resized:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(str(destination))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = load_model(str(args.config), str(args.checkpoint), device=args.device)
    records = []
    combined_by_view = {"agentview": [], "backview": []}

    for task_dir in sorted(path for path in args.video_root.iterdir() if path.is_dir()):
        task_index = _task_index(task_dir)
        for view in ("agentview", "backview"):
            source_path = task_dir / "vlsa_I" / "0" / "{}.png".format(view)
            image_source, image_tensor = load_image(str(source_path))
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption="chocolate pudding",
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )
            raw = Image.fromarray(np.asarray(image_source, dtype=np.uint8), mode="RGB")
            detections = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                normalized = [float(value) for value in box.tolist()]
                detections.append(
                    {
                        "phrase": phrase,
                        "confidence": float(logit.max().item()),
                        "box_cxcywh_normalized": normalized,
                        "box_xyxy_pixels": _xyxy_pixels(normalized, raw.width, raw.height),
                    }
                )
            if not detections:
                raise RuntimeError(
                    "GroundingDINO returned no chocolate-pudding detection for task {} {}".format(
                        task_index, view
                    )
                )

            annotated = _draw_detection(raw, detections)
            annotated_path = args.output_root / "task_{}_{}_bbox.png".format(task_index, view)
            combined_path = args.output_root / "task_{}_{}_view_and_bbox.png".format(
                task_index, view
            )
            annotated.save(str(annotated_path))
            _side_by_side(raw, annotated, task_index, view).save(str(combined_path))
            combined_by_view[view].append(combined_path)
            records.append(
                {
                    "task_index": task_index,
                    "view": view,
                    "prompt": "chocolate pudding",
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                    "source": str(source_path),
                    "annotated": str(annotated_path),
                    "view_and_bbox": str(combined_path),
                    "detections": detections,
                }
            )
            print(
                "task={} view={} detections={} max_confidence={:.4f}".format(
                    task_index,
                    view,
                    len(detections),
                    max(item["confidence"] for item in detections),
                ),
                flush=True,
            )

    for view, paths in combined_by_view.items():
        paths.sort()
        _contact_sheet(
            paths,
            args.output_root / "all_4_{}_view_and_bbox.png".format(view),
        )
    manifest = {
        "device": args.device,
        "model_config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "records": sorted(records, key=lambda record: (record["task_index"], record["view"])),
    }
    (args.output_root / "detections.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

