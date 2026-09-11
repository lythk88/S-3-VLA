"""Obstacle-name selection for safety geometry perception.

``simulator`` reproduces the historical deterministic prompt mapping.
``vlsa-api`` sends agentview and the original VLSA prompt to GLM-4.5V.
API answers are used raw: no simulator lookup, canonicalization, or fallback.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
import requests


VLSA_OBSTACLE_NAMES = (
    "yellow rectangular book",
    "blue moka pot",
    "red mug",
    "white storage box",
    "black wine bottle",
    "red milk carton",
)
VLSA_LONG_OBSTACLE_NAME = "gray rectangular binder"


@dataclass(frozen=True)
class ObstacleSelection:
    name: str
    source: str
    prompt: str | None = None
    model: str | None = None
    latency_s: float | None = None
    simulator_mapping_used: bool = False

    def record(self) -> dict:
        return asdict(self)


def build_original_vlsa_prompt(instruction: str, task_suite_name: str) -> str:
    """Return the prompt string from the original VLSA implementation."""
    prefer_list = list(VLSA_OBSTACLE_NAMES)
    if "long" in task_suite_name:
        prefer_list.append(VLSA_LONG_OBSTACLE_NAME)
    return (
        f"The robot must follow this instruction: {instruction}. Based on both the "
        "instruction and the image, identify exactly one non-robot object that is most "
        "likely to obstruct the robot's motion during task execution. You must output a "
        "uniquely identifiable obstacle name including both color and object type, "
        f"preferably from this list when applicable: {prefer_list}. Output only the object "
        "name, with no additional words."
    )


def simulator_obstacle_prompt(obstacle_instance: str) -> str:
    """Return the historical deterministic GroundingDINO prompt."""
    prompt_by_type = {
        "milk": "red milk carton",
        "moka_pot": "blue moka pot",
        "red_coffee_mug": "red mug",
        "white_storage_box": "white storage box",
        "wine_bottle": "black wine bottle",
        "yellow_book": "gray rectangular book",
    }
    obstacle_type = obstacle_instance.rsplit("_obstacle_", 1)[0]
    if obstacle_type.endswith("_small"):
        obstacle_type = obstacle_type[: -len("_small")]
    return prompt_by_type.get(obstacle_type, obstacle_type.replace("_", " "))


def _chat_completions_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def _usable_api_key() -> str:
    key = os.environ.get("ZAI_API_KEY")
    if key:
        return key
    # This experiment environment stores the complete id.secret token under
    # this historical variable name.
    legacy_complete_key = os.environ.get("ZHIPUAI_API_KEY_SECRET")
    if legacy_complete_key and legacy_complete_key.count(".") == 1:
        return legacy_complete_key
    key = os.environ.get("ZHIPUAI_API_KEY")
    if key:
        return key
    raise ValueError("Set ZAI_API_KEY or ZHIPUAI_API_KEY for VLSA API selection")


def select_with_original_vlsa_api(
    image: Image.Image | np.ndarray,
    instruction: str,
    task_suite_name: str,
    *,
    model: str,
    base_url: str,
    timeout_s: float,
) -> ObstacleSelection:
    """Call GLM with the original VLSA request and return its raw answer."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    encoded_image = io.BytesIO()
    image.convert("RGB").save(encoded_image, format="JPEG", quality=95)
    image_data = base64.b64encode(encoded_image.getvalue()).decode("ascii")
    prompt = build_original_vlsa_prompt(instruction, task_suite_name)
    payload = {
        "model": model,
        "messages": [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
                "role": "user",
            }
        ],
        "temperature": 0.1,
        "top_p": 0.1,
        "thinking": {"type": "enabled"},
    }
    started = time.perf_counter()
    response = requests.post(
        _chat_completions_endpoint(base_url),
        headers={
            "Authorization": f"Bearer {_usable_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    if not isinstance(answer, str):
        raise TypeError(f"Unexpected GLM response content: {type(answer).__name__}")
    answer = (
        answer.replace("<|begin_of_box|>", "")
        .replace("<|end_of_box|>", "")
        .strip()
    )
    if not answer:
        raise RuntimeError("GLM obstacle selector returned an empty answer")
    return ObstacleSelection(
        name=answer,
        source="original_vlsa_glm_api",
        prompt=prompt,
        model=model,
        latency_s=time.perf_counter() - started,
        simulator_mapping_used=False,
    )


def select_obstacle_name(
    *,
    mode: str,
    image: Image.Image | np.ndarray,
    instruction: str,
    task_suite_name: str,
    simulator_instance: str,
    prompt_override: str | None,
    api_model: str,
    api_base_url: str,
    api_timeout_s: float,
) -> ObstacleSelection:
    """Resolve one obstacle prompt under an explicit selection protocol."""
    if mode == "vlsa-api":
        if prompt_override:
            raise ValueError("prompt_override is incompatible with vlsa-api mode")
        return select_with_original_vlsa_api(
            image,
            instruction,
            task_suite_name,
            model=api_model,
            base_url=api_base_url,
            timeout_s=api_timeout_s,
        )
    if mode != "simulator":
        raise ValueError(f"Unsupported obstacle selector mode: {mode!r}")
    if prompt_override:
        return ObstacleSelection(name=prompt_override, source="prompt_override")
    return ObstacleSelection(
        name=simulator_obstacle_prompt(simulator_instance),
        source="simulator_instance_mapping",
        simulator_mapping_used=True,
    )
