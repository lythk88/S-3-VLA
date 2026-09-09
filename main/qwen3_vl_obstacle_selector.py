"""Local Qwen3-VL obstacle naming with the original VLSA prompt."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image


BASE_OBSTACLE_NAMES = (
    "gray rectangular book",
    "blue moka pot",
    "red mug",
    "white storage box",
    "black wine bottle",
    "red milk carton",
)
LONG_ONLY_OBSTACLE_NAME = "gray rectangular binder"


def build_vlsa_obstacle_prompt(instruction: str, task_suite_name: str) -> str:
    """Return the prompt text used by the original VLSA API implementation."""
    prefer_list = list(BASE_OBSTACLE_NAMES)
    if "long" in task_suite_name:
        prefer_list.append(LONG_ONLY_OBSTACLE_NAME)
    return (
        f"The robot must follow this instruction: {instruction}. Based on both the "
        "instruction and the image, identify exactly one non-robot object that is most "
        "likely to obstruct the robot's motion during task execution. You must output a "
        "uniquely identifiable obstacle name including both color and object type, "
        f"preferably from this list when applicable: {prefer_list}. Output only the object "
        "name, with no additional words."
    )


def canonicalize_obstacle_answer(answer: str) -> str | None:
    """Map a free-form VLM answer to a SafeLIBERO obstacle class for scoring."""
    text = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    if "binder" in text:
        return LONG_ONLY_OBSTACLE_NAME
    if "moka" in text or "coffee pot" in text:
        return "blue moka pot"
    if "wine bottle" in text or ("black" in text and "bottle" in text):
        return "black wine bottle"
    if "milk" in text and any(word in text for word in ("carton", "box", "container")):
        return "red milk carton"
    if "storage" in text and any(word in text for word in ("box", "container", "bin")):
        return "white storage box"
    if "book" in text:
        return "gray rectangular book"
    if "mug" in text and ("red" in text or "coffee" in text):
        return "red mug"
    return None


@dataclass(frozen=True)
class ObstaclePrediction:
    raw_answer: str
    canonical_answer: str | None
    prompt: str
    latency_s: float


class Qwen3VLObstacleSelector:
    """Lazy local inference wrapper for Qwen3-VL-8B-Instruct."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
        max_new_tokens: int = 24,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        load_kwargs = {
            "local_files_only": local_files_only,
            "torch_dtype": torch.bfloat16,
        }
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **load_kwargs,
        ).eval()
        self.model.to(torch.device(device))

    def predict(
        self,
        image: Image.Image | np.ndarray,
        instruction: str,
        task_suite_name: str,
    ) -> ObstaclePrediction:
        import torch

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        image = image.convert("RGB")
        prompt_text = build_vlsa_obstacle_prompt(instruction, task_suite_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat],
            images=[image],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        answer = answer.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()
        return ObstaclePrediction(
            raw_answer=answer,
            canonical_answer=canonicalize_obstacle_answer(answer),
            prompt=prompt_text,
            latency_s=time.perf_counter() - started,
        )


_DEFAULT_SELECTOR: Qwen3VLObstacleSelector | None = None


def get_default_qwen3_vl_obstacle_selector() -> Qwen3VLObstacleSelector:
    """Return one process-wide selector configured through environment variables."""
    global _DEFAULT_SELECTOR
    if _DEFAULT_SELECTOR is None:
        model_id = os.environ.get("QWEN3_VL_MODEL", "/dev/shm/Qwen3-VL-8B-Instruct")
        device = os.environ.get("QWEN3_VL_DEVICE", "cpu")
        local_only = os.environ.get("QWEN3_VL_LOCAL_FILES_ONLY", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        _DEFAULT_SELECTOR = Qwen3VLObstacleSelector(
            model_id,
            device=device,
            local_files_only=local_only,
        )
    return _DEFAULT_SELECTOR
