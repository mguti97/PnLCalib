import sys
from dataclasses import field

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from shapely.geometry import LineString, Polygon



def plane_from_P(P, cam_pos, principal_point):
    def is_invertible(a):
        # return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
        return np.linalg.cond(a) < 1 / np.finfo(a.dtype).eps

    if not any(np.isnan(P.flatten())):
        H = np.delete(P, 2, axis=1)
        pp = np.array([principal_point[0], principal_point[1], 1.])

        if is_invertible(H):
            pp_proj = np.linalg.inv(H) @ pp
        else:
            pp_proj = np.linalg.pinv(H) @ pp
        pp_proj /= pp_proj[-1]
        plane_vector = pp_proj - cam_pos

        return plane_vector, cam_pos
    else:
        return None, None


def plane_from_H(H, cam_pos, principal_point):
    def is_invertible(a):
        # return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
        return np.linalg.cond(a) < 1 / np.finfo(a.dtype).eps

    if not any(np.isnan(H.flatten())):
        pp = np.array([principal_point[0], principal_point[1], 1.])

        if is_invertible(H):
            pp_proj = np.linalg.inv(H) @ pp
        else:
            pp_proj = np.linalg.pinv(H) @ pp
        pp_proj /= pp_proj[-1]
        plane_vector = pp_proj - cam_pos

        return plane_vector, cam_pos
    else:
        return None, None


def is_in_front_of_plane(point, plane_normal, plane_point):
    return np.dot(point - plane_point, plane_normal) > 0


def line_plane_intersection(p1, p2, plane_normal, plane_point, epsilon=0.5):
    points_clipped = []
    p1 = np.array(p1)
    p2 = np.array(p2)

    p1_f = is_in_front_of_plane(p1, plane_normal, plane_point)
    p2_f = is_in_front_of_plane(p2, plane_normal, plane_point)
    p_f = [p1_f, p2_f]

    if not p1_f and not p2_f:
        return points_clipped

    if (p1_f and p2_f):
        return [p1, p2]

    for count, p in enumerate([p1, p2]):
        if p_f[count]:
            points_clipped.append(p)
        else:
            # Line direction vector
            line_dir = p2 - p1

            # Check if the line and plane are parallel
            denom = np.dot(plane_normal, line_dir)
            if np.isclose(denom, 0):
                # Line and plane are parallel (no intersection or line is within the plane)
                continue

            # Calculate the value of t
            t = np.dot(plane_normal, (plane_point - p1)) / denom

            # Find the intersection point
            intersection_point = p1 + t * line_dir
            intersection_point += epsilon * plane_normal / np.linalg.norm(plane_normal)
            points_clipped.append(intersection_point)

    return points_clipped


def get_opt_vector(pos, rot):
    rot_vector, _ = cv2.Rodrigues(rot)

    return np.concatenate((pos, rot_vector.ravel()))

def get_opt_seq_vector(mtx, k1, pos, rot):
    x_focal_length = np.array([mtx[0, 0]])
    y_focal_length = np.array([mtx[1, 1]])
    k1 = np.array([k1])
    rot_vector, _ = cv2.Rodrigues(rot)

    return np.concatenate((x_focal_length, y_focal_length, k1, pos, rot_vector[:,0]))


def vector_to_mtx(vector, mtx):
    x_focal_length = mtx[0, 0]
    y_focal_length = mtx[1, 1]
    principal_point = (mtx[0, 2], mtx[1, 2])
    position_meters = vector[:3]

    rot_vector = np.array(vector[3:])
    rotation, _ = cv2.Rodrigues(rot_vector)

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P

def vector_seq_to_mtx(vector, principal_point):
    fx = vector[0]
    fy = vector[1]
    k = vector[2]
    t = np.array(vector[3:6])
    rot_vector = np.array(vector[6:])
    rotation, _ = cv2.Rodrigues(rot_vector)

    It = np.eye(4)[:-1]
    It[:, -1] = -t
    Q = np.array([[fx, 0, principal_point[0]],
                  [0, fy, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P


def from_vector(vector, principal_point):
    fx = vector[0]
    fy = vector[1]
    k = vector[2]
    t = np.array(vector[3:6])
    rot_vector = np.array(vector[6:])
    rotation, _ = cv2.Rodrigues(rot_vector)

    It = np.eye(4)[:-1]
    It[:, -1] = -t
    Q = np.array([[fx, 0, principal_point[0]],
                  [0, fy, principal_point[1]],
                  [0, 0, 1]])

    return rotation, t, Q, k

def point_to_line_distance(l1, l2, p):
    A = (l2[1] - l1[1])
    B = (l2[0] - l1[0])
    C = l2[0] * l1[1] - l2[1] * l1[0]

    num = (A * p[0] - B * p[1] + C)
    den = np.sqrt(A ** 2 + B ** 2)

    if den > 0:
        return num / den
    else:
        return 0


def generate_grid_points(image_shape, step=20, margin=10):
    h, w = image_shape
    xs = np.arange(margin, w - margin, step)
    ys = np.arange(margin, h - margin, step)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return grid.reshape(-1, 1, 2)  # Nx1x2

def ray_from_xy(xy, K, R, t, k1=0.0, k2=0.0):
    """
    Compute the ray from the camera center through the image point (x, y),
    correcting for radial distortion using coefficients k1 and k2.

    Args:
        xy: (2,) array_like containing pixel coordinates [x, y] in the image.
        K: (3, 3) ndarray representing the camera intrinsic matrix.
        R: (3, 3) ndarray representing the camera rotation matrix.
        t: (3,) ndarray representing the camera center in world coordinates.
        k1, k2 (float): Radial distortion coefficients (default 0).

    Returns:
        origin: (3,) ndarray representing the camera center in world coordinates.
        direction: (3,) unit ndarray representing the direction of the ray in world coordinates.
    """
    # Step 1: Convert the pixel coordinate to normalized camera coordinates
    p = np.array([xy[0], xy[1], 1.0])
    p_norm = np.linalg.inv(K) @ p  # shape (3,)
    x_d, y_d = p_norm[0], p_norm[1]

    # Step 2: Apply inverse radial distortion (approximation)
    r2 = x_d**2 + y_d**2
    factor = 1 + k1 * r2 + k2 * (r2**2)
    x_undist = x_d / factor
    y_undist = y_d / factor

    # Step 3: Construct direction vector in camera coordinates
    d_cam = np.array([x_undist, y_undist, 1.0])

    # Step 4: Rotate direction into world coordinates
    direction = R.T @ d_cam
    direction /= np.linalg.norm(direction)

    # Step 5: Camera center is already in world coordinates
    origin = t
    return origin, direction

def intersection_over_plane(o, d):
    """
    args:
        o: (3,) origin of the ray
        d: (3,) direction of the ray

    returns:
        intersection: (3,) intersection point
    """
    # solve the x and y where z = 0
    t_scale = -o[2] / d[2]
    return o + t_scale * d

def build_field_mask(image_shape, K, R, t):
    h, w = image_shape
    field_corners = np.array([[-105/2, -68/2, 0], [105/2, -68/2, 0], [105/2, 68/2, 0], [-105/2, 68/2, 0]])

    It = np.eye(4)[:-1]
    It[:, -1] = -t
    P = K @ (R @ It)

  # 3x4 extrinsics
    X_hom = np.hstack([field_corners, np.ones((len(field_corners), 1))])  # Nx4
    x_proj = (P @ X_hom.T).T  # Nx3
    x_proj /= x_proj[:, 2:3]  # normalize
    field_poly = x_proj[:, :2]  # (4, 2)

    # Step 3: Draw polygon on mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [field_poly.astype(np.int32)], 1)
    return 1-mask

def build_person_mask(image_shape, bboxes, scale_percent=1):
    h, w = image_shape
    mask = np.ones((h, w), dtype=np.uint8)

    for (x, y, w_box, h_box) in bboxes:
        cx = x + w_box / 2
        cy = y + h_box / 2

        new_w = w_box * (1 + scale_percent)
        new_h = h_box * (1 + scale_percent)

        x1 = int(max(cx - new_w / 2, 0))
        y1 = int(max(cy - new_h / 2, 0))
        x2 = int(min(cx + new_w / 2, w))
        y2 = int(min(cy + new_h / 2, h))

        mask[y1:y2, x1:x2] = 0

    return mask


def compute_optical_flow(prev_frame, curr_frame, K, R, t, bboxes=None, k1=0.0, k2=0.0, step=150):
    """
    Compute optical flow correspondences from a fixed grid, filtered by a field mask and bounding boxes,
    and backproject rays to 3D points on the Z=0 plane using given camera parameters.

    Args:
        prev_frame (np.ndarray): Grayscale image at time t-1.
        curr_frame (np.ndarray): Grayscale image at time t.
        K (np.ndarray): 3x3 intrinsic matrix for frame t-1.
        R (np.ndarray): 3x3 rotation matrix for frame t-1.
        t (np.ndarray): (3,) camera center in world coordinates.
        bboxes (list of tuples): [(x, y, w, h), ...] in pixel coords, for exclusion.
        k1, k2 (float): Radial distortion coefficients (default 0).
        step (int): Grid spacing in pixels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (pts_curr, pts_prev, X_3D), where:
            - pts_curr: Nx2 tracked points in current frame.
            - pts_prev: Nx2 original grid points in previous frame.
            - X_3D: Nx3 corresponding 3D points on the field (from rays in prev frame).
    """
    h, w = prev_frame.shape[:2]

    # Step 1: Generate a fixed grid of image coordinates
    xs = np.arange(step // 2, w, step)
    ys = np.arange(step // 2, h, step)
    grid_pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)

    # Step 2: Create field mask (1 = valid, 0 = out of field)
    field_mask = build_field_mask((h, w), K, R, t)

    # Step 3: Create bbox exclusion mask (1 = valid, 0 = inside bbox)
    if bboxes:
        bbox_mask = build_person_mask((h, w), bboxes, scale_percent=1)
        final_mask = field_mask * bbox_mask
    else:
        final_mask = field_mask


    # Step 5: Filter grid points using combined mask
    grid_pts_int = grid_pts.astype(np.int32)
    valid_mask = [
        0 <= y < h and 0 <= x < w and final_mask[y, x] > 0
        for x, y in grid_pts_int
    ]
    grid_pts = grid_pts[valid_mask].reshape(-1, 1, 2)  # Nx1x2

    if len(grid_pts) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3))

    # Step 6: Track points using LK optical flow
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, grid_pts, None, winSize=(10, 10))

    # Step 7: Filter valid tracks
    status = status.flatten()
    pts_prev = grid_pts[status == 1][:, 0]   # Nx2
    pts_curr = curr_pts[status == 1][:, 0]   # Nx2

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    # ax1.imshow(prev_frame)
    # ax2.imshow(prev_frame)
    # for pt in pts_prev:
    #     ax2.scatter(pt[0], pt[1], s=1, c='r')
    # im = ax3.imshow(final_mask)
    # fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    # fig.savefig('test.png', dpi=fig.dpi)
    # sys.exit()

    # Step 8: Backproject to 3D field points
    X_3D = []
    for pt in pts_prev:
        o, d = ray_from_xy(pt, K, R, t, k1, k2)
        X = intersection_over_plane(o, d)
        X_3D.append(X)

    X_3D = np.array(X_3D)  # Nx3

    return pts_curr, X_3D


def compute_semantic_optical_flow(prev_frame, curr_frame, K, R, t, bboxes=None, k1=0.0, k2=0.0, step=2):
    line_world_coords_3D = [[[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                            [[16.5, 13.84, 0.], [0., 13.84, 0.]], [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                            [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                            [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]],
                            [[0., 30.34, 0.], [0., 30.34, -2.44]], [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                            [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
                            [[52.5, 0., 0.], [52.5, 68, 0.]], [[0., 68., 0.], [105., 68., 0.]],
                            [[0., 0., 0.], [0., 68., 0.]],
                            [[105., 0., 0.], [105., 68., 0.]], [[0., 0., 0.], [105., 0., 0.]],
                            [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                            [[5.5, 43.16, 0.], [5.5, 24.84, 0.]], [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                            [[99.5, 43.16, 0.], [105., 43.16, 0.]], [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                            [[99.5, 24.84, 0.], [105., 24.84, 0.]]]

    line_world_coords_3D = [[[x1 - 52.5, y1 - 34, z1], [x2 - 52.5, y2 - 34, z2]] for [[x1, y1, z1], [x2, y2, z2]] in
                            line_world_coords_3D]

    h, w = prev_frame.shape[:2]

    grid_pts = []
    grid_pts_3D = []
    It = np.eye(4)[:-1]
    It[:, -1] = -t
    P = K @ (R @ It)
    plane_normal, plane_point = plane_from_P(P, t, K[:-1,-1])
    for line in line_world_coords_3D:
        pti, ptf = line
        p = line_plane_intersection(pti, ptf, plane_normal, plane_point)
        if len(p) > 0:
            pti, ptf = p
            vec = ptf - pti
            length = np.linalg.norm(vec)
            direction = vec / length
            num_steps = int(np.floor(length / step)) + 1
            points = np.array([pti + i * step * direction for i in range(num_steps)])

            for pt in points:
                proj_pt, _ = cv2.projectPoints(pt, R, -R@t, K, np.array([k1, k2, 0., 0.]))
                if 0 <= proj_pt[0,0][0] <= w and 0 <= proj_pt[0,0][1] <= h:
                    grid_pts.append(proj_pt[0,0])
                    grid_pts_3D.append(pt)

    grid_pts = np.array(grid_pts)
    grid_pts_3D = np.array(grid_pts_3D)

    if bboxes:
        bbox_mask = build_person_mask((h, w), bboxes, scale_percent=1)

    grid_pts_int = grid_pts.astype(np.int32)
    valid_mask = [
        0 <= y < h and 0 <= x < w and bbox_mask[y, x] > 0
        for x, y in grid_pts_int
    ]
    grid_pts = grid_pts[valid_mask].astype(np.float32).reshape(-1, 1, 2)  # Nx1x2
    grid_pts_3D = grid_pts_3D[valid_mask].astype(np.float32)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, grid_pts, None, winSize=(10, 10))

    if status is not None:
        status = status.flatten()
        pts_curr = curr_pts[status == 1][:, 0]  # Nx2
        grid_pts_3D = grid_pts_3D[status == 1]

        return pts_curr, grid_pts_3D
    else:
        return np.array([]), np.array([])
