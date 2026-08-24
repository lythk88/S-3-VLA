# GroundingDINO yellow rectangular book — 16 SafeLIBERO tasks

- View: agentview only
- Suites: spatial=4, goal=4, object=4, long=4
- Target localized: 7/16 (43.75%)
- Status counts: correct=4, correct_with_extra_boxes=3, wrong_localization=9
- Placement validation: stable and surface-supported in every scene; zero task-object, robot, or gripper contacts

## Per-suite performance

| Suite | Localized | Total | Rate | Mean best IoU |
|---|---:|---:|---:|---:|
| safelibero_spatial | 1 | 4 | 25.00% | 0.158 |
| safelibero_goal | 3 | 4 | 75.00% | 0.706 |
| safelibero_object | 1 | 4 | 25.00% | 0.256 |
| safelibero_long | 2 | 4 | 50.00% | 0.467 |

## Per-task results

| Suite | Task | Status | Detections | Best IoU | Confidence |
|---|---:|---|---:|---:|---:|
| safelibero_spatial | 0 | wrong_localization | 1 | 0.000 | 0.654 |
| safelibero_spatial | 1 | correct_with_extra_boxes | 2 | 0.634 | 0.548 |
| safelibero_spatial | 2 | wrong_localization | 1 | 0.000 | 0.708 |
| safelibero_spatial | 3 | wrong_localization | 1 | 0.000 | 0.638 |
| safelibero_goal | 0 | correct_with_extra_boxes | 2 | 0.941 | 0.435 |
| safelibero_goal | 1 | correct | 1 | 0.941 | 0.404 |
| safelibero_goal | 2 | wrong_localization | 1 | 0.000 | 0.545 |
| safelibero_goal | 3 | correct | 1 | 0.941 | 0.504 |
| safelibero_object | 0 | wrong_localization | 1 | 0.000 | 0.806 |
| safelibero_object | 1 | wrong_localization | 1 | 0.000 | 0.823 |
| safelibero_object | 2 | wrong_localization | 4 | 0.080 | 0.680 |
| safelibero_object | 3 | correct_with_extra_boxes | 2 | 0.945 | 0.567 |
| safelibero_long | 0 | correct | 1 | 0.939 | 0.780 |
| safelibero_long | 1 | wrong_localization | 1 | 0.000 | 0.709 |
| safelibero_long | 2 | correct | 1 | 0.929 | 0.806 |
| safelibero_long | 3 | wrong_localization | 1 | 0.000 | 0.670 |
