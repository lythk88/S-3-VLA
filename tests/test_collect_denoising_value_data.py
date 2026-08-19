from unittest import mock

import numpy as np

from main.collect_denoising_value_data import _default_max_steps
from main.collect_denoising_value_data import _execute_branch


class _BaseEnv:
    def __init__(self):
        self.timestep = 7
        self.cur_time = 0.35
        self.done = False


class _Env:
    def __init__(self):
        self.env = _BaseEnv()

    def regenerate_obs_from_state(self, state):
        return state

    def step(self, action):
        if self.env.done:
            raise ValueError("executing action in terminated episode")
        self.env.timestep += 1
        self.env.cur_time += 0.05
        self.env.done = self.env.timestep >= 10
        return {}, 0.0, False, {}


def test_suite_default_horizons():
    assert _default_max_steps("safelibero_spatial") == 220
    assert _default_max_steps("safelibero_object") == 300
    assert _default_max_steps("safelibero_long") == 550


def test_branch_rollouts_do_not_consume_nominal_episode_clock():
    env = _Env()
    actions = np.zeros((3, 7), dtype=np.float32)
    clearance = mock.patch(
        "main.collect_denoising_value_data._clearance_and_collision",
        return_value=(0.03, False),
    )

    with clearance:
        for _ in range(4):
            values, collided = _execute_branch(
                env, np.zeros(1), actions, set(), set(), 0.03, 3
            )
            np.testing.assert_allclose(values, [0.03, 0.03, 0.03])
            assert not collided

    assert env.env.timestep == 7
    assert env.env.cur_time == 0.35
    assert not env.env.done


def test_branch_clock_is_restored_when_step_raises():
    env = _Env()
    env.env.timestep = 9
    actions = np.zeros((2, 7), dtype=np.float32)

    with mock.patch(
        "main.collect_denoising_value_data._clearance_and_collision",
        return_value=(0.03, False),
    ), mock.patch.object(env, "step", side_effect=RuntimeError("simulator failure")):
        try:
            _execute_branch(env, np.zeros(1), actions, set(), set(), 0.03, 2)
        except RuntimeError as error:
            assert str(error) == "simulator failure"
        else:
            raise AssertionError("expected simulator failure")

    assert env.env.timestep == 9
    assert env.env.cur_time == 0.35
    assert not env.env.done
