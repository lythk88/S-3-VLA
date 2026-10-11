# Success-Guided Safe Flow Matching

Inference-time safety for a frozen **π0.5** flow-matching VLA on **SafeLIBERO**.
During denoising, each intermediate action chunk is corrected by a small convex
program that combines

- a **success direction** `g_succ = ∇_A log V_φ(Z_τ, A, τ)` from a learned,
  denoising-conditioned success critic, and
- **discrete-time CBF constraints** `h_j ≥ (1-γ) h_{j-1}` on the predicted
  clearance between the protected body (gripper ellipsoid, plus the carried
  object once grasped) and a primitive fitted to the obstacle point cloud.

```
min_δ  ‖δ‖² − β_succ g_succᵀδ
s.t.   ĥ_j + J_j δ ≥ (1−γ)(ĥ_{j−1} + J_{j−1} δ),   j = 1..H_q
       ‖δ‖₂ ≤ r_trust,   a_min ⪯ A + S_qᵀδ ⪯ a_max
```

If the program is infeasible, a safety-first fallback first maximizes the
worst-case linearized barrier (LP), then picks the most success-aligned
correction that keeps that safety level (QP). The corrected chunk is fed back
into the flow, and the remaining prefix is re-projected with `β_succ = 0` before
each executed action (closed-loop, receding horizon).

## Method → code

| Method component | File | Entry points |
|---|---|---|
| Frozen π0.5 flow + access to intermediate denoising states | `openpi/src/openpi/models/pi0_denoising_trace.py` | `sample_actions_with_denoising_trace`, `install()` |
| Success critic `V_φ` (hidden + action tokens, Fourier time embedding, transformer, learned-query pooling) | `Success-critic/success_critic_model.py` | `SuccessCritic` |
| Critic training (rollout-level stratified split, progress-weighted class-balanced BCE, Brier model selection) | `Success-critic/train_success_critic.py` | CLI |
| Trained critic | `Success-critic/success_critic_v1/` | `best_model.pt`, `training_manifest.json`, `history.json` |
| Obstacle primitive fit `P̂` (box / cylinder / sphere, penalized surface error) | `main/primitive_fitting.py` | `fit_best_primitive`, `fit_primitive_candidates` |
| RGB-D back-projection, GroundingDINO detection, gripper ellipsoid | `main/utils.py` | `get_point_cloud`, `fit_ellipse` |
| Which obstacle to protect (simulator name or VLSA prompt to a VLM API) | `main/obstacle_selection.py` | `select_obstacle_name` |
| Grasp trigger `χ_k` and carried-object box (support-plane removal, clustering, trimmed extents) | `main/grasp_rgbd.py` | `gripper_width`, `select_grasped_object_points` |
| State predictor `F_κ`, signed clearance, barriers `h_j`, linearization `J_j` | `openpi/src/openpi/policies/action_expert_qp.py` | `rollout_eef_trajectory`, `primitive_obstacle_gap`, `trajectory_barriers`, `linearize_barriers` |
| Success-guided QCQP, trust-radius escalation, two-stage fallback | `openpi/src/openpi/policies/action_expert_qp.py` | `solve_action_expert_qp`, `project_action_chunk_with_adaptive_radius`, `solve_maximin_barrier_qp`, `refine_action_chunk_for_success` |
| Steering at denoising times (`A⁺ = A + S_qᵀδ*`, then resume the flow) | `openpi/src/openpi/policies/action_expert_guidance_policy.py` | `ActionExpertGuidancePolicy` |
| Policy server | `scripts/serve_action_expert_guidance_policy.py` | CLI |
| Closed-loop rollout: perception, compound body, pre-execution re-projection, re-query after grasp | `main/main_aegis.py` | `--use-action-expert-guidance` |
| Provenance (source / checkpoint hashes per run) | `main/run_manifest.py` | `write_run_manifest` |

`main/main_aegis.py` also keeps two baselines: nominal π0.5
(`--disable-safety-layer` without `--use-action-expert-guidance`) and the
original AEGIS/VLSA CBF safety layer (the default when `--disable-safety-layer`
is not set).

### Implementation details not spelled out in the method text

- The objective also contains a deviation weight `λ_dev` (`--action-expert-lambda-deviation`, 1 in the reported runs).
- Corrections act only on the translational action channels. Rotation and gripper channels keep their nominal values (`--action-expert-translation-only-execution`).
- Obstacles can be extruded upward by a one-sided Minkowski padding (`--action-expert-obstacle-top-padding-m`, 0.02 m in the reported runs).
- The QP is applied at denoising times τ ∈ {0.3, 0.1} (`--action-expert-times`).

## Layout

```
main/            rollout client, perception, primitive fitting, grasp estimation (+ unit tests)
openpi/          trimmed fork of Physical-Intelligence/openpi with the guidance policy and QP
Success-critic/  critic model, training script, trained checkpoint (v1)
scripts/         policy server and the two reproducible evaluation runs
```

## Setup

Requirements not included in this repo:

| Dependency | Used for |
|---|---|
| SafeLIBERO checkout + `config.yaml` | Simulator and tasks (`LIBERO_SOURCE_ROOT`, `LIBERO_CONFIG_PATH`) |
| π0.5 LIBERO checkpoint (`pi05_libero`) | Frozen policy (`POLICY_CHECKPOINT_DIR`) |
| GroundingDINO SwinT-OGC weights + config | Obstacle detection |
| Python env for openpi (`openpi/pyproject.toml`, `uv`) | Policy server (JAX) + critic (PyTorch) |
| Python env for the rollout client (`main/requirements.txt`) | LIBERO / MuJoCo / cvxpy |
| `ZAI_API_KEY` (or `ZHIPUAI_API_KEY`) | Only for `--obstacle-selector vlsa-api` |

The scripts look for the runtime under `RUNTIME_ROOT` (default: this repo if it
contains `.venv/`, otherwise `/workspace/safe-flow-matching`). Every path can
be overridden through environment variables. See the top of each script.

## Running the evaluations

```bash
# SafeLIBERO Object, level I, tasks 0-3, episodes 0-9, simulator obstacle names
bash scripts/run_3shape_critic_on_object_I_ep0to9_40.sh

# SafeLIBERO Spatial, level II, VLM (VLSA prompt) obstacle selection
bash scripts/run_best3shape_compound_spatial_II_vlsa_api_200.sh
```

Each script checks that the critic checkpoint's sha256 matches the expected
value. It then starts the guidance policy server, runs one `main_aegis.py`
worker per task, and writes logs and per-episode summaries to
`$OUT` (default `/dev/shm/<RUN_NAME>`, symlinked into `results/`).

To run the server alone:

```bash
python scripts/serve_action_expert_guidance_policy.py \
  --checkpoint-dir <pi05_libero> --critic-run-dir Success-critic/success_critic_v1 \
  --torch-device cuda --port 8011
```

## Success critic

### Training

```bash
python Success-critic/train_success_critic.py \
  --trace-root <traces> --output-dir Success-critic/success_critic_vX --device cuda
```

The trainer expects one compressed archive per rollout at
`<trace-root>/<task>/<condition>/<episode>_success_trace.npz`. The
`<task>/<condition>` pair is the stratum used by the rollout-level split.
Required keys:

| Key | Shape | Meaning |
|---|---|---|
| `success` | `()` | Rollout task success (label `y_i`) |
| `episode_steps` | `()` | Episode length, used for the progress `ρ` |
| `chunk_start_steps` | `(C,)` | Control step at which each chunk was queried |
| `denoising_hidden_states` | `(C, T, H, 1024)` | Action-expert hidden tokens per denoising step |
| `denoising_noisy_actions` | `(C, T, H, 32)` | Intermediate action chunk `A_τ` |
| `denoising_times` | `(C, T)` | Denoising time `τ` |

`C` = chunks per rollout, `T` = recorded denoising steps, `H` = 10 action tokens.

`openpi/src/openpi/policies/denoising_trace_policy.py` is the server-side
wrapper that returns these arrays (via the `__debug_denoising_trace__`
request). **The rollout-side collector that wrote the v1 traces is not part of
this repository.**

### v1 checkpoint (validation, 62 held-out rollouts)

| Metric | Per sample | Per rollout |
|---|---|---|
| AUC | 0.844 | 0.983 |
| Brier | 0.159 | 0.074 |
| Accuracy @ 0.5 | 0.778 | 0.935 |

Train: 183 rollouts (124 success, 59 failure). Best epoch: 3.
sha256 `a78d5e99…f044b`.

## Tests

```bash
cd main && python -m pytest -q          # needs robosuite for grasp/obstacle tests
cd openpi && pytest src/openpi/policies/action_expert_qp_test.py
```

Known issue: 4 tests in `main/primitive_fitting_test.py` currently fail
(`ValueError: truth value of an array ... is ambiguous` in
`primitive_fitting.py:487`).

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (π0.5), the
[VLSA / AEGIS](https://vlsa-aegis.github.io/) codebase, and the
[SafeLIBERO](https://huggingface.co/datasets/THURCSCT/SafeLIBERO) benchmark.
The vendored openpi code keeps its own license (`openpi/LICENSE`).
