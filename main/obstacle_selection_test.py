from unittest import mock

import numpy as np

from obstacle_selection import build_original_vlsa_prompt, select_obstacle_name


def test_original_vlsa_prompt_is_exact() -> None:
    prompt = build_original_vlsa_prompt(
        "put the bowl on the plate",
        "safelibero_goal",
    )
    assert prompt == (
        "The robot must follow this instruction: put the bowl on the plate. Based on both "
        "the instruction and the image, identify exactly one non-robot object that is most "
        "likely to obstruct the robot's motion during task execution. You must output a "
        "uniquely identifiable obstacle name including both color and object type, "
        "preferably from this list when applicable: ['yellow rectangular book', 'blue moka "
        "pot', 'red mug', 'white storage box', 'black wine bottle', 'red milk carton']. "
        "Output only the object name, with no additional words."
    )


def test_api_mode_does_not_use_simulator_mapping() -> None:
    predicted = mock.Mock(
        name="selection",
        source="original_vlsa_glm_api",
        prompt="prompt",
        model="glm-4.5v",
        latency_s=1.0,
        simulator_mapping_used=False,
    )
    with mock.patch(
        "obstacle_selection.select_with_original_vlsa_api",
        return_value=predicted,
    ) as api, mock.patch(
        "obstacle_selection.simulator_obstacle_prompt",
        side_effect=AssertionError("simulator mapping must not run"),
    ):
        result = select_obstacle_name(
            mode="vlsa-api",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            instruction="pick up the black bowl",
            task_suite_name="safelibero_spatial",
            simulator_instance="yellow_book_obstacle_1",
            prompt_override=None,
            api_model="glm-4.5v",
            api_base_url="https://example.invalid/v4/",
            api_timeout_s=10.0,
        )
    assert result is predicted
    api.assert_called_once()
