#!/usr/bin/env python3
"""Run the existing evaluator with a randomized BDDL reset.

This module deliberately lives outside the existing evaluator. It substitutes an
environment class whose ``set_init_state`` keeps the state sampled by ``reset``.
The original SafeLIBERO source and its fixed test initial states remain unchanged.
"""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main"))
sys.path.insert(0, str(ROOT / "safelibero"))

import main_aegis  # noqa: E402
import tyro  # noqa: E402


class RandomizedResetEnv(main_aegis.OffScreenRenderEnv):
    """Ignore the fixed test state and retain the BDDL-randomized reset state."""

    def reset(self):
        self._randomized_observation = super().reset()
        return self._randomized_observation

    def set_init_state(self, init_state):
        del init_state
        return self._randomized_observation


def main() -> None:
    main_aegis.OffScreenRenderEnv = RandomizedResetEnv
    main_aegis.eval_libero(tyro.cli(main_aegis.Args))


if __name__ == "__main__":
    main()
