# Best three-shape compound SMCBF

This branch is a frozen, minimal method snapshot for the 200-rollout
SafeLIBERO Spatial-II evaluation. It intentionally excludes prior ablations,
analysis utilities, visualizers, unrelated benchmarks, and vendored simulator
copies.

## Frozen evaluation configuration

- Obstacle name: GLM-4.5V API using the original VLSA agentview prompt. The raw
  answer is passed to GroundingDINO without simulator mapping or fallback.
- Obstacle geometry: one geometric fit selection from AABB, cylinder, and
  sphere; no QP-based shape selector.
- Safety geometry: gripper ellipsoid plus grasp-conditioned carried-object
  compound box.
- Obstacle padding: upper world-Z face only.
- Execution: pi0.5's full nominal 7D action is retained, including rotation and
  gripper commands. The receding closed-loop QP corrects translation only,
  with a shorter QP prefix executed one action at a time inside each policy
  chunk.
- Success guidance: `success_critic_v1`.

The rollout script expects the simulator, GroundingDINO, Python environment,
and checkpoint assets in a separate runtime checkout (default:
`/workspace/safe-flow-matching`). This keeps the method branch small while the
committed source and manifest identify every method component.

Run the frozen evaluation with:

```bash
bash scripts/run_best3shape_compound_spatial_II_vlsa_api_200.sh
```
