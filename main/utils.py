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
def get_point_cloud(image, depth, env, view, TEXT_PROMPT, model, save_path):
    # from robosuite.utils.camera_utils import get_real_depth_map
    from groundingdino.util.inference import load_image, predict, annotate
    depth = get_real_depth_map(env.sim, depth)
    # import cv2
    # CONFIG_PATH = "GroundingDINO/GroundingDINO_SwinT_OGC.py"    # Config file included in source code
    # CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"   # Downloaded weights file
    # The simulator environment may use a CUDA build that predates the assigned
    # GPU architecture. Keep this configurable so detection can safely fall
    # back to CPU while the policy server continues to use the GPU.
    DEVICE = os.environ.get("GROUNDINGDINO_DEVICE", "cuda")
    BOX_TRESHOLD = 0.35     # Bounding box threshold given by source code
    TEXT_TRESHOLD = 0.25    # Text threshold for key attributes given by source code

    # model = load_model(CONFIG_PATH, CHECKPOINT_PATH)


    import matplotlib
    matplotlib.use('Agg')          
    import matplotlib.pyplot as plt
    plt.imsave(str(save_path / f"{view}.png"), image)
    import cv2
    IMAGE_PATH = str(save_path /f"{view}.png")
    image_source, image_cv = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model,
        image=image_cv,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE,
    )
    try:
        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )
        cv2.imwrite(str(save_path / f"annotated_ {view}_image.jpg"), annotated_frame)
    except (AttributeError, TypeError) as exc:
        # Annotation is diagnostic-only. Some installed supervision releases do
        # not expose the API expected by GroundingDINO, but the predicted boxes
        # below remain valid inputs to the safety geometry pipeline.
        print(f"Skipping GroundingDINO annotation for {view}: {exc}")
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy
    import torch
    image = image[::-1, ::-1]
    depth = depth[::-1, ::-1].squeeze()
    # 2. Check if any object is detected
    if boxes.shape[0] > 0:

        h, w, _ = image.shape
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        size_tensor = torch.tensor([w, h, w, h], device=boxes.device)
        pixel_boxes = boxes_xyxy * size_tensor
        first_box = pixel_boxes[0].cpu().numpy().astype(int)
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
    import open3d as o3d

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
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.array(
        pcd.cluster_dbscan(eps=0.0001, min_points=50, print_progress=False)
    )

    # If valid clusters exist (labels>=0), take only the largest cluster
    if labels.max() >= 0:
        largest = np.bincount(labels[labels >= 0]).argmax()
        mask = (labels == largest)
        pts = np.asarray(pcd.points)[mask]




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
    from zai import ZhipuAiClient
    import base64
    import matplotlib
    matplotlib.use('Agg')  # ✅ Crucial: No GUI rendering
    import matplotlib.pyplot as plt
    import time
    api_key = ""

    if api_key is None or api_key == "":
        raise ValueError("Please enter api_key")
    client = ZhipuAiClient(api_key=api_key)  # Fill in your own APIKey
    plt.imsave("obstacle_detection.png", image)

    image_path="obstacle_detection.png"
    t0 = time.time()
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    print("large model is thinking...")
    prefer_list = ['yellow rectangular book', 'blue moka pot',  'red mug', 'white storage box', 'black wine bottle', 'red milk carton']
    if "long" in task_suite_name:
        prefer_list.append('gray rectangular binder')
    response = client.chat.completions.create(
        model="glm-4.5v",  # Fill in the model name to call
        messages=[
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"The robot must follow this instruction: {instruction}. Based on both the instruction and the image, identify exactly one non-robot object that is most likely to obstruct the robot's motion during task execution. You must output a uniquely identifiable obstacle name including both color and object type, preferably from this list when applicable: {prefer_list}. Output only the object name, with no additional words."
                    }
                ],
                "role": "user",

            }
        ],
        temperature=0.1, # Reduce randomness for more deterministic answers
        top_p=0.1,
        thinking={
            "type":"enabled"
        }
    )
    clean_output = response.choices[0].message.content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    print(clean_output)  # Output: gray cube
    t1 = time.time()
    print(f"large model thinking time: {t1-t0}s")
    return clean_output
