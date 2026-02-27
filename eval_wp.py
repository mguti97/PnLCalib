import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from matplotlib.patches import Polygon
from scipy.spatial.transform import Rotation

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib_wp import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict


def list_npz_files(directory):
    return [f for f in os.listdir(directory)
            if f.endswith('.npz') and os.path.isfile(os.path.join(directory, f))]

def list_jpg_files_with_prefix(directory, prefix):
    return [f for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith('.jpg') and os.path.isfile(os.path.join(directory, f))]


def compute_translation_error(t_est, t_gt):
    """
    Computes translation error (Euclidean distance) between two translation vectors.

    Args:
        t_est (np.ndarray): Estimated translation vector (3,).
        t_gt (np.ndarray): Ground truth translation vector (3,).

    Returns:
        float: Translation error (Euclidean distance).
    """
    t_est = np.asarray(t_est).reshape(3)
    t_gt = np.asarray(t_gt).reshape(3)
    return np.linalg.norm(t_est - t_gt)


def compute_orientation_error(R_current, R_desired=np.eye(3)):
    """
    Computes orientation error between current and desired rotation matrices using quaternions.

    Args:
        R_current (np.ndarray): 3x3 current rotation matrix.
        R_desired (np.ndarray): 3x3 desired rotation matrix (default is identity).

    Returns:
        error_quat (np.ndarray): Quaternion [x, y, z, w] representing the orientation error.
        angle_error (float): Angle (in radians) of the rotation error.
        axis_error (np.ndarray): 3D axis of rotation.
    """
    # Convert rotation matrices to quaternions
    q_current = Rotation.from_matrix(R_current)
    q_desired = Rotation.from_matrix(R_desired)

    # Compute relative rotation (error quaternion)
    q_error = q_desired.inv() * q_current

    # Quaternion [x, y, z, w]
    error_quat = q_error.as_quat()

    # Get angle-axis representation
    angle_error = q_error.magnitude()
    axis_error = q_error.as_rotvec()
    if angle_error > 1e-8:
        axis_error = axis_error / angle_error  # Normalize axis
    else:
        axis_error = np.array([0.0, 0.0, 0.0])  # No significant rotation

    #return error_quat, angle_error, axis_error
    return angle_error


def calib_dict_from_file(mtx, dist, frame_num):
    frame_mtx, frame_dist = mtx[frame_num], dist[frame_num]
    calib_dict = {"intrinsics": frame_mtx, "distortion": frame_dist}
    return calib_dict

def inference(cam, frame_num, frame, mtx, dist, model, model_l, kp_threshold, line_threshold, pnl_refine):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    calib_dict = calib_dict_from_file(mtx, dist, frame_num)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(args.device)
    b, c, h, w = frame.size()

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

    cam.update(calib_dict, kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

    return final_params_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--calib_path", type=str, help="Path to camera calibration file")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CPU or CUDA device index")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))

    loaded_state = torch.load(args.weights_kp, map_location=args.device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(args.device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=args.device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(args.device)
    model_l.eval()

    transform2 = T.Resize((540, 960))

    cam = FramebyFrameCalib(iwidth=1920, iheight=1080, denormalize=True)

    seqs = list_npz_files(args.calib_path)
    e_R_list, e_t_list = [], []
    for seq in seqs:
        seq_name = seq.split(".")[0]
        print(seq_name)
        calib = np.load(args.calib_path + "/" + seq)
        mtx, dist = calib["K"], calib["k"]

        imgs = list_jpg_files_with_prefix(args.input_path, seq_name)
        imgs.sort()

        if imgs:
            e_R_seq, e_t_seq = [], []
            for img in tqdm(imgs):
                frame = cv2.imread(args.input_path + "/" + img, cv2.IMREAD_COLOR)
                frame_num = int(img.split("_")[-1].split(".")[0])
                final_params_dict = inference(
                    cam, frame_num, frame, mtx, dist,
                    model, model_l, args.kp_threshold, args.line_threshold,
                    args.pnl_refine
                )

                cam_params = final_params_dict["cam_params"]
                R = np.array(cam_params["rotation_matrix"])
                t = np.array(cam_params["position_meters"])
                t = -R @ t

                e_R = compute_orientation_error(R, calib["R"][frame_num])
                e_t = compute_translation_error(t, calib["t"][frame_num])
                e_R_seq.append(e_R)
                e_t_seq.append(e_t)

            print(
                f"Sequence {seq}: MRE = {round(np.mean(e_R_seq), 7)} degrees \t MTE = {round(np.mean(e_t_seq), 7)} m")
            e_R_list += e_R_seq
            e_t_list += e_t_seq


    print("\nOverall stats")
    print(f"MRE = {round(np.mean(e_R_list), 3)} degrees")
    print(f"MTE = {round(np.mean(e_t_list), 3)} m")





