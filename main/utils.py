import pathlib
import logging

import numpy as np
from scipy.spatial.transform import Rotation as R
from robosuite.utils.camera_utils import get_real_depth_map
import cvxpy as cp
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



############ CBF Related Functions ##################
def rot3(euler):
    """3D rotation matrix from ZYX Euler angles (yaw, pitch, roll)."""
    roll, pitch, yaw = euler
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]])
    Ry = np.array([[cy, 0, sy],
                   [0,  1, 0],
                   [-sy,0, cy]])
    Rx = np.array([[1, 0, 0],
                   [0, cx,-sx],
                   [0, sx, cx]])
    return Rx @ Ry @ Rz

def quat_R(quat_mj):
    quat_scipy = [quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]]  # [x,y,z,w]
    r = R.from_quat(quat_scipy)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def quat_euler(quat_mj):
    quat_scipy = [quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]]  # [x,y,z,w]
    r = R.from_quat(quat_scipy)
    euler = r.as_euler('XYZ', degrees=False)
    return euler

def vector_hat(v):
    """hat operator: R^3 -> so(3)"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def project_matrix(z):
    """Compute (I - z z^T)."""
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)

    Q = np.diag(Q_diag)
    Qbar = R @ Q @ R.T
    Qbar_inv = np.linalg.inv(Qbar)
    z = z_dir / (np.linalg.norm(z_dir) + 1e-12)

    nvec = (Qbar_inv @ z).ravel()
    d = -(1 + z.T @ Qbar_inv @ p)
    a, b, c = nvec

    lim = plane_size
    grid = np.linspace(-lim, lim, npts)

    # choose best projection axis
    if abs(c) >= abs(a) and abs(c) >= abs(b):
        xx, yy = np.meshgrid(grid, grid)
        zz = (-a*xx - b*yy - d) / (c + 1e-12)
        return xx, yy, zz
    elif abs(b) >= abs(a):
        xx, zz = np.meshgrid(grid, grid)
        yy = (-a*xx - c*zz - d) / (b + 1e-12)
        return xx, yy, zz
    else:
        yy, zz = np.meshgrid(grid, grid)
        xx = (-b*yy - c*zz - d) / (a + 1e-12)
        return xx, yy, zz


def compute_h_ij(p_i, Q_i_diag, R_i,
                 p_j, Q_j_diag, R_j,
                 z_ij):
    # Calculate the value of CBF h
    # Shape matrices
    Q_i = np.diag(Q_i_diag)
    Q_j = np.diag(Q_j_diag)

    # World-shape matrices
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T

    # Inverses
    Qbar_i_inv = np.linalg.inv(Qbar_i)

    # Direction
    z = z_ij / np.linalg.norm(z_ij)

    # Compute numerator and denominator
    term1 = np.linalg.norm(Qbar_j @ Qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ Qbar_i_inv @ z
    denom = np.linalg.norm(Qbar_i_inv @ z)

    h_ij = (-term1 + term2 - 1.0) / denom
    return h_ij

def compute_h_coeffs_3d(p_i, Q_i_diag, R_i,
                        p_j, Q_j_diag, R_j,
                        z,
                        eps=1e-10):
    # Calculate relevant coefficients in CBF derivatives
    # build matrices
    Q_i = np.diag(Q_i_diag); Q_j = np.diag(Q_j_diag)
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    Qbar_i_inv2 = Qbar_i_inv @ Qbar_i_inv
    Qbar_j2 = Qbar_j @ Qbar_j

    z = z / (np.linalg.norm(z)+eps)
    a_vec = Qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = Qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = (1.0 - (p_j - p_i).T @ a_vec + term1)

    # eta_row and xi_row (paper)
    eta_row = - (1.0 / denom) * (z.T @ Qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ Qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ Qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ Qbar_i_inv @ Qbar_j2 @ Qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    # zeta_tilde and nu_tilde (only need zeta_tilde)
    tmp1 = z.T @ Qbar_i_inv2 @ vector_hat(z)
    left_vec = (z.T @ Qbar_i_inv @ Qbar_j2)
    Ja_vec = vector_hat(a_vec)
    tmp2 = left_vec @ (Ja_vec - Qbar_i_inv @ vector_hat(z))
    part_a = (p_j - p_i).T @ Qbar_i_inv @ vector_hat(z)
    part_b = z.T @ Qbar_i_inv @ vector_hat(p_j - p_i)
    tmp3 = part_a + part_b
    zeta_tilde = rho * (1.0 / (denom**3 + eps)) * tmp1 + (1.0 / sigma) * tmp2 + (1.0 / denom) * tmp3

    # a_v corresponds to coefficients on world-frame velocity R_i v_i: eta_row @ (R_i)
    a_v = (eta_row @ R_i).ravel()
    a_omega = zeta_tilde @ R_i

    # a_uz: mu_row @ (I - z z^T)
    a_uz = (mu_row @ project_matrix(z)).ravel()

    # compute h for alpha(h)
    h = compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z)

    return a_v, a_omega, a_uz, h, mu_row   # mu_row is dh/d z_ij

############## Perception Related Functions ##############
def overlay_gripper_ellipsoid_on_rgb(
    image,
    env,
    view,
    center,
    rotation,
    radii,
    save_path,
):
    """Project the gripper safety ellipsoid into a saved RGB camera view."""
    import cv2
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    azimuth = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    elevation = np.linspace(0.0, np.pi, 48)
    unit_surface = np.stack(
        (
            np.outer(np.cos(azimuth), np.sin(elevation)),
            np.outer(np.sin(azimuth), np.sin(elevation)),
            np.outer(np.ones_like(azimuth), np.cos(elevation)),
        ),
        axis=-1,
    ).reshape(-1, 3)
    world_surface = (unit_surface * radii) @ rotation.T + center
    world_points = np.vstack((world_surface, center))
    homogeneous = np.column_stack((world_points, np.ones(len(world_points))))
    world_to_pixels = get_camera_transform_matrix(
        env.sim,
        view,
        camera_height=height,
        camera_width=width,
    )
    camera_homogeneous = homogeneous @ world_to_pixels.T
    in_front = camera_homogeneous[:, 2] > 1e-6
    projected = camera_homogeneous[in_front, :2] / camera_homogeneous[in_front, 2:3]

    # main_aegis saves both simulator image axes reversed. Under that exact
    # convention, camera (u, v) maps to saved RGB pixel (width - 1 - u, v).
    pixels = np.column_stack((width - 1 - projected[:, 0], projected[:, 1]))
    finite = np.all(np.isfinite(pixels), axis=1)
    pixels = pixels[finite]
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    pixels = np.rint(pixels).astype(np.int32)
    if len(pixels) < 3:
        raise RuntimeError(f"Could not project gripper ellipsoid into {view}")

    surface_pixels = pixels[:-1]
    center_pixel = tuple(int(value) for value in pixels[-1])
    hull = cv2.convexHull(surface_pixels.reshape(-1, 1, 2))
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, hull, color=(35, 105, 255))
    result = cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0)
    cv2.polylines(result, [hull], isClosed=True, color=(0, 55, 255), thickness=5)
    cv2.drawMarker(
        result,
        center_pixel,
        color=(0, 55, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=4,
    )
    cv2.putText(
        result,
        "blue = projected gripper safety ellipsoid",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 55, 255),
        3,
        cv2.LINE_AA,
    )
    from PIL import Image

    Image.fromarray(result).save(save_path)


def get_point_cloud(
    image,
    depth,
    env,
    view,
    TEXT_PROMPT,
    model,
    save_path,
    *,
    save_diagnostics=True,
    selection_reference_world=None,
    candidate_top_k=5,
    vlm_classifier=None,
):
    # from robosuite.utils.camera_utils import get_real_depth_map
    depth = get_real_depth_map(env.sim, depth)
    # import cv2
    # CONFIG_PATH = "GroundingDINO/GroundingDINO_SwinT_OGC.py"    # Config file included in source code
    # CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"   # Downloaded weights file
    # The simulator environment may use a CUDA build that predates the assigned
    # GPU architecture. Keep this configurable so detection can safely fall
    # back to CPU while the policy server continues to use the GPU.
    DEVICE = os.environ.get("GROUNDINGDINO_DEVICE", "cuda")
    # Keep several candidates for the carried-object classifier.  A low
    # detector threshold is intentional here: GroundingDINO's top score is
    # often a visually similar bottle (e.g. the wine bottle in BBQ episodes).
    BOX_TRESHOLD = 0.15 if vlm_classifier is not None else 0.35
    TEXT_TRESHOLD = 0.25    # Text threshold for key attributes given by source code

    # model = load_model(CONFIG_PATH, CHECKPOINT_PATH)


    import tempfile

    from PIL import Image

    if save_diagnostics:
        image_path = save_path / f"{view}.png"
    else:
        temporary_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temporary_image.close()
        image_path = pathlib.Path(temporary_image.name)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(image_path)
    import cv2
    IMAGE_PATH = str(image_path)
    import torch
    transformer_backend = isinstance(model, dict) and model.get("backend") == "transformers"
    if transformer_backend:
        from PIL import Image
        from PIL import ImageDraw

        image_source = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))
        caption = TEXT_PROMPT.strip().rstrip(".") + "."
        inputs = model["processor"](
            images=image_source, text=caption, return_tensors="pt"
        ).to(model["device"])
        with torch.no_grad():
            outputs = model["model"](**inputs)
        detection = model["processor"].post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            target_sizes=[image_source.shape[:2]],
        )[0]
        pixel_boxes = detection["boxes"]
        detection_scores = detection.get("scores", torch.zeros(len(pixel_boxes)))
        boxes = pixel_boxes
        annotated_image = Image.fromarray(image_source)
        drawing = ImageDraw.Draw(annotated_image)
        labels = detection.get("labels", [TEXT_PROMPT] * len(pixel_boxes))
        scores = detection.get("scores", [None] * len(pixel_boxes))
        for box, label, score in zip(pixel_boxes, labels, scores):
            x1, y1, x2, y2 = [int(value) for value in box.detach().cpu().tolist()]
            drawing.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
            score_text = "" if score is None else f" {float(score):.3f}"
            drawing.text(
                (x1 + 3, max(0, y1 - 12)),
                f"{label}{score_text}",
                fill=(255, 0, 0),
                stroke_width=1,
                stroke_fill=(255, 255, 255),
            )
        if not len(pixel_boxes):
            drawing.text((5, 5), f"No detection: {TEXT_PROMPT}", fill=(255, 0, 0))
        if save_diagnostics:
            annotated_image.save(save_path / f"annotated_{view}.png")
    else:
        from groundingdino.util.inference import annotate, load_image, predict

        image_source, image_cv = load_image(IMAGE_PATH)
        boxes, logits, phrases = predict(
            model=model,
            image=image_cv,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE,
        )
        detection_scores = logits
        try:
            annotated_frame = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases,
            )
            if save_diagnostics:
                cv2.imwrite(str(save_path / f"annotated_{view}.png"), annotated_frame)
        except (AttributeError, TypeError) as exc:
            print(f"Skipping GroundingDINO annotation for {view}: {exc}")
    if not save_diagnostics:
        image_path.unlink(missing_ok=True)
    image = image[::-1, ::-1]
    depth = depth[::-1, ::-1].squeeze()
    # 2. Check if any object is detected
    if boxes.shape[0] > 0:

        h, w, _ = image.shape
        if not transformer_backend:
            from groundingdino.util.box_ops import box_cxcywh_to_xyxy
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            size_tensor = torch.tensor([w, h, w, h], device=boxes.device)
            pixel_boxes = boxes_xyxy * size_tensor
        # Rank candidates by detector score and retain only the requested top-k.
        # This prevents a low-confidence tail of spurious boxes from reaching
        # the VLM while preserving alternatives for visual disambiguation.
        if len(pixel_boxes) > candidate_top_k:
            score_array = torch.as_tensor(detection_scores).detach().cpu().numpy()
            keep = np.argsort(-score_array)[:candidate_top_k]
            pixel_boxes = pixel_boxes[keep]
            detection_scores = torch.as_tensor(score_array[keep])
        selected_box_index = 0
        if vlm_classifier is not None and len(pixel_boxes) > 0:
            # Work in the same 180-degree-rotated image used for point-cloud
            # extraction.  The classifier returns the candidate rank.
            candidate_paths = []
            for rank, box in enumerate(pixel_boxes):
                bx1, by1, bx2, by2 = [int(round(v)) for v in box.detach().cpu().tolist()]
                cx1, cx2 = max(0, w - 1 - bx2), min(w, w - 1 - bx1)
                cy1, cy2 = max(0, h - 1 - by2), min(h, h - 1 - by1)
                crop = image[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                candidate_path = save_path / f"candidate_{view}_{rank+1}.png"
                Image.fromarray(crop.astype(np.uint8)).save(candidate_path)
                candidate_paths.append(candidate_path)
            if candidate_paths:
                selected_box_index = int(vlm_classifier(candidate_paths, TEXT_PROMPT))
                selected_box_index = max(0, min(selected_box_index, len(pixel_boxes) - 1))
                logging.info(
                    "VLM selected GroundingDINO candidate %d/%d for %r in %s",
                    selected_box_index + 1, len(pixel_boxes), TEXT_PROMPT, view,
                )
        if selection_reference_world is not None and len(pixel_boxes) > 1 and vlm_classifier is None:
            from robosuite.utils.camera_utils import get_camera_transform_matrix

            world_to_pixels = get_camera_transform_matrix(
                env.sim,
                view,
                camera_height=h,
                camera_width=w,
            )
            reference = np.append(
                np.asarray(selection_reference_world, dtype=np.float64), 1.0
            )
            projected = world_to_pixels @ reference
            if projected[2] > 1e-9 and np.all(np.isfinite(projected)):
                raw_pixel = projected[:2] / projected[2]
                saved_pixel = np.array([w - 1 - raw_pixel[0], raw_pixel[1]])
                box_array = pixel_boxes.detach().cpu().numpy()
                inside = (
                    (box_array[:, 0] <= saved_pixel[0])
                    & (saved_pixel[0] <= box_array[:, 2])
                    & (box_array[:, 1] <= saved_pixel[1])
                    & (saved_pixel[1] <= box_array[:, 3])
                )
                centers = 0.5 * (box_array[:, :2] + box_array[:, 2:])
                distances = np.linalg.norm(centers - saved_pixel[None, :], axis=1)
                distances[~inside] += float(max(h, w))
                selected_box_index = int(np.argmin(distances))
                logging.info(
                    "GroundingDINO %r selected box %d/%d nearest projected "
                    "world target %s in %s",
                    TEXT_PROMPT,
                    selected_box_index,
                    len(pixel_boxes),
                    np.asarray(selection_reference_world),
                    view,
                )
        first_box = pixel_boxes[selected_box_index].cpu().numpy().astype(int)
        x1, y1, x2, y2 = first_box
        xmin = w - 1 - x2
        xmax = w - 1 - x1
        ymin = h - 1 - y2
        ymax = h - 1 - y1

        # Limit to image range
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        cropped_image_rgb = image[ymin:ymax, xmin:xmax]
        cropped_image_depth = depth[ymin:ymax, xmin:xmax]
        if cropped_image_rgb.size > 0:
            cropped_image_bgr = cv2.cvtColor(cropped_image_rgb, cv2.COLOR_RGB2BGR)
            # save_path = "cropped_a_view_milk_carton.jpg"
            # cv2.imwrite(save_path, cropped_image_bgr)
        else:
            print(f"❌ Crop failed, invalid bounding box coordinates: [{xmin}, {ymin}, {xmax}, {ymax}]")
            return np.array([[]])

    else:
        print("❌ GroundingDINO detected no objects, unable to crop.")
        return np.array([[]])



    h_full, w_full = image.shape[0], image.shape[1]
  
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_camera_intrinsic_matrix
    K_inv = np.linalg.inv(get_camera_intrinsic_matrix(env.sim, view, h_full, w_full))
    # print(K_inv)

    T_cam_to_world = get_camera_extrinsic_matrix(env.sim, view)
    # print(T_cam_to_world)

    v_full, u_full = np.indices((h_full, w_full))
    v_full = (h_full - 1) - v_full

    cropped_u = u_full[ymin:ymax, xmin:xmax]
    cropped_v = v_full[ymin:ymax, xmin:xmax]


    u_flat = cropped_u.flatten()
    v_flat = cropped_v.flatten()
    depth_flat = cropped_image_depth.flatten() # (cropped_depth already exists)
    colors_flat = cropped_image_rgb.reshape(-1, 3) # (cropped_rgb already exists)


    pixels_homo = np.stack([u_flat, v_flat, np.ones_like(u_flat)], axis=0)
    points_cam = K_inv @ pixels_homo * depth_flat

    points_cam_homo = np.vstack([points_cam, np.ones_like(depth_flat)])
    points_world_homo = T_cam_to_world @ points_cam_homo
    points = points_world_homo[:3, :].T
    
    return points

def filtering_points(pts, task_suite_name):
    import numpy as np
    from sklearn.cluster import DBSCAN

    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)

# --- Step 1: XYZ Range Filtering ---
    if "spatial" in task_suite_name or "goal" in task_suite_name:
    # table scene
        keep = (
            (pts[:, 2] > 0.92) & (pts[:, 2] < 1.5) &   # Z
            (pts[:, 0] > -0.3) & (pts[:, 0] < 0.3) &  # X
            (pts[:, 1] > -0.3) & (pts[:, 1] < 0.3)    # Y
        )
    elif "object" in task_suite_name:
    # floor scene
        keep = (
            (pts[:, 2] > 0.05) & (pts[:, 2] < 0.5) &   # Z
            (pts[:, 0] > -0.3) & (pts[:, 0] < 0.3) &  # X
            (pts[:, 1] > -0.3) & (pts[:, 1] < 0.3)    # Y
        )
    elif "long" in task_suite_name:
    #living_room_table
        keep = (
            (pts[:, 2] > 0.43) & (pts[:, 2] < 0.8) &   # Z
            (pts[:, 0] > -0.3) & (pts[:, 0] < 0.3) &  # X
            (pts[:, 1] > -0.3) & (pts[:, 1] < 0.3)    # Y
        )

    pts = pts[keep]

    if len(pts) == 0:
        return pts

     # --- Step 1.5: Remove the 20% of points farthest from the centroid ---
    center = pts.mean(axis=0)                  # Centroid
    dist = np.linalg.norm(pts - center, axis=1)  # Distance from each point to centroid

    # Sort distances from small to large
    keep_count = int(len(pts) * 0.8)  # Keep nearest 80%
    sorted_indices = np.argsort(dist)
    pts = pts[sorted_indices[:keep_count]]

    if len(pts) == 0:
        return pts  

    # --- Step 2: DBSCAN Clustering ---
    labels = DBSCAN(eps=0.0001, min_samples=50, n_jobs=1).fit_predict(pts)

    # If valid clusters exist (labels>=0), take only the largest cluster
    if labels.max() >= 0:
        largest = np.bincount(labels[labels >= 0]).argmax()
        mask = (labels == largest)
        pts = pts[mask]




    return pts
def mvee_cvxpy(P):
    # Minimum Volume Enclosing Ellipsoid (MVEE) fitting
    N, d = P.shape

    M = cp.Variable((d, d), PSD=True)  
    g = cp.Variable(d)


    objective = cp.Minimize(-cp.log_det(M))


    constraints = [cp.norm(M @ P[i] - g) <= 1 for i in range(N)]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        print("SCS solver failed, trying default solver...")
        prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"MVEE convex optimization failed. Status: {prob.status}")

    # Extract results
    M_opt = M.value
    g_opt = g.value

    # Calculate center c and matrix A
    # c = M^-1 * g
    # A = M^T * M
    c = np.linalg.solve(M_opt, g_opt)
    A = M_opt.T @ M_opt

    return c, A


def plot_points_ellipse(points, center, R, axes_diag, save_path="examples/libero/results/ellipse_plot.png"):
    # Plot obstacle point cloud
    import matplotlib
    matplotlib.use('Agg')  # ✅ Crucial: No GUI rendering
    import matplotlib.pyplot as plt
  
    # -------------------------
    # 1. Point Cloud
    # -------------------------
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # -------------------------
    # 2. Ellipsoid Sampling
    # -------------------------
    a, b, c = axes_diag
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))

    ellipsoid = np.stack([x, y, z], axis=-1)
    ellipsoid_world = ellipsoid @ R.T + center.reshape(1, 1, 3)

    ex = ellipsoid_world[:, :, 0]
    ey = ellipsoid_world[:, :, 1]
    ez = ellipsoid_world[:, :, 2]

    # -------------------------
    # 3. Plotting
    # -------------------------
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Point cloud
    ax.scatter(X, Y, Z, s=4, c='blue', alpha=0.7)

    # Ellipsoid (semi-transparent)
    ax.plot_surface(ex, ey, ez, color='red', alpha=0.5, rstride=2, cstride=2)

    # Axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    # Axis equal scaling
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
        ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
        ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

    set_axes_equal(ax)
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.4, 0.4])


    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # print(f"✅ Image saved to: {save_path}")


def plot_gripper_obstacle_ellipsoids(
    gripper_center,
    gripper_rotation,
    gripper_radii,
    obstacle_center,
    obstacle_rotation,
    obstacle_radii,
    *,
    points=None,
    save_path="gripper_obstacle_ellipsoids.png",
):
    """Plot both safety ellipsoids and report whether the obstacle contains the gripper."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gripper_center = np.asarray(gripper_center, dtype=np.float64)
    gripper_rotation = np.asarray(gripper_rotation, dtype=np.float64)
    gripper_radii = np.asarray(gripper_radii, dtype=np.float64)
    obstacle_center = np.asarray(obstacle_center, dtype=np.float64)
    obstacle_rotation = np.asarray(obstacle_rotation, dtype=np.float64)
    obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 32)
    unit = np.stack(
        (
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ),
        axis=-1,
    )

    def surface(center, rotation, radii):
        return (unit * radii) @ rotation.T + center

    gripper_surface = surface(gripper_center, gripper_rotation, gripper_radii)
    obstacle_surface = surface(obstacle_center, obstacle_rotation, obstacle_radii)
    gripper_in_obstacle = (gripper_surface.reshape(-1, 3) - obstacle_center) @ obstacle_rotation
    maximum_obstacle_level = float(
        np.max(np.sum(np.square(gripper_in_obstacle / obstacle_radii), axis=1))
    )
    obstacle_covers_gripper = maximum_obstacle_level <= 1.0 + 1e-6

    displacement = obstacle_center - gripper_center
    center_distance = float(np.linalg.norm(displacement))
    if center_distance > 1e-12:
        direction = displacement / center_distance

        def support(rotation, radii):
            shape = rotation @ np.diag(np.square(radii)) @ rotation.T
            return float(np.sqrt(max(float(direction @ shape @ direction), 0.0)))

        directional_gap = center_distance - support(gripper_rotation, gripper_radii) - support(
            obstacle_rotation, obstacle_radii
        )
    else:
        directional_gap = -float(np.max(gripper_radii) + np.max(obstacle_radii))

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(*np.moveaxis(gripper_surface, -1, 0), color="royalblue", alpha=0.35)
    axis.plot_surface(*np.moveaxis(obstacle_surface, -1, 0), color="crimson", alpha=0.35)
    axis.scatter(*gripper_center, color="navy", s=45, label="gripper ellipsoid")
    axis.scatter(*obstacle_center, color="darkred", s=45, label="obstacle MVEE")
    axis.plot(
        [gripper_center[0], obstacle_center[0]],
        [gripper_center[1], obstacle_center[1]],
        [gripper_center[2], obstacle_center[2]],
        color="black",
        linestyle="--",
    )
    if points is not None and len(points):
        points = np.asarray(points)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color="gray", alpha=0.2)
    combined = np.vstack((gripper_surface.reshape(-1, 3), obstacle_surface.reshape(-1, 3)))
    midpoint = 0.5 * (np.min(combined, axis=0) + np.max(combined, axis=0))
    radius = 0.55 * float(np.max(np.ptp(combined, axis=0)))
    axis.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    axis.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    axis.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_zlabel("world z (m)")
    axis.set_title(
        f"Obstacle covers gripper: {obstacle_covers_gripper} | directional gap: {directional_gap:.4f} m"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=220)
    plt.close(figure)
    return {
        "obstacle_covers_gripper": obstacle_covers_gripper,
        "directional_gap": directional_gap,
        "maximum_obstacle_level_on_gripper_surface": maximum_obstacle_level,
    }


def fit_ellipse(pts, plot=False, save_path="examples/libero/results/ellipse_plot.png"):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts) # Extract convex hull
    hull_pts = pts[hull.vertices]
    center, A = mvee_cvxpy(hull_pts)
    eigvals, eigvecs = np.linalg.eigh(A)  # A is symmetric positive definite
    eigvals = np.clip(eigvals, 1e-15, None)
    axes = 1.0 / np.sqrt(eigvals)  # Semi-axis lengths a,b,c
    sort_idx = np.argsort(axes)[::-1]
    axes = axes[sort_idx]
    R = eigvecs[:, sort_idx]  # Corresponding eigenvectors must also be reordered
    S = axes
    if plot:
        plot_points_ellipse(pts, center, R, S, str(save_path/"ellipse_plot.png"))
    return center, R, S

def obstacle_detection(image, instruction, task_suite_name):
    """Select the obstacle with local Qwen3-VL using the original VLSA prompt."""
    from qwen3_vl_obstacle_selector import get_default_qwen3_vl_obstacle_selector

    prediction = get_default_qwen3_vl_obstacle_selector().predict(
        image,
        instruction,
        task_suite_name,
    )
    logging.info(
        "Qwen3-VL obstacle answer=%r canonical=%r latency=%.3fs",
        prediction.raw_answer,
        prediction.canonical_answer,
        prediction.latency_s,
    )
    return prediction.raw_answer
