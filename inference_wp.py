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

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib_wp import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict



def inference(cam, frame_num, frame, mtx, dist, model, model_l, kp_threshold, line_threshold, pnl_refine):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    calib_dict = calib_dict_from_file(mtx, dist, frame_num)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(device)
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


def calib_dict_from_file(mtx, dist, frame_num):
    frame_mtx, frame_dist = mtx[frame_num], dist[frame_num]
    calib_dict = {"intrinsics": frame_mtx, "distortion": frame_dist}
    return calib_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--calib_path", type=str, help="Path to camera calibration file")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.1, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.1, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CPU or CUDA device index")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video.")
    args = parser.parse_args()

    calib_path = args.calib_path
    input_path = args.input_path
    model_kp = args.weights_kp
    model_line = args.weights_line
    pnl_refine = args.pnl_refine
    save_path = args.save_path
    device = args.device
    kp_threshold = args.kp_threshold
    line_threshold = args.line_threshold

    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform2 = T.Resize((540, 960))

    cam = FramebyFrameCalib(iwidth=1920, iheight=1080, denormalize=True)
    vid_paths = [vid for end in ['*.mp4', '*.mov'] for vid in Path(input_path).glob(end)]

    if vid_paths:
        for vid_path in vid_paths:
            cap = cv2.VideoCapture(str(vid_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_name = vid_path.name.split(".")[0]

            if os.path.exists(calib_path + vid_name + ".npz"):
                vid_calib = np.load(calib_path + vid_name + ".npz")
                mtx, dist = vid_calib["K"], vid_calib["k"]

                R_array = np.zeros((total_frames, 3, 3))
                t_array = np.zeros((total_frames, 3))

                frame_num = 0
                with tqdm(total=total_frames, desc=f"Processing {vid_name}", unit="frame") as pbar:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if frame_num + 1 > mtx.shape[0]:
                            break
                        if not ret:
                            frame_num += 1
                            pbar.update(1)
                            break

                        final_params_dict = inference(
                            cam, frame_num, frame, mtx, dist,
                            model, model_l, kp_threshold, line_threshold,
                            pnl_refine
                        )

                        cam_params = final_params_dict["cam_params"]
                        R = cam_params["rotation_matrix"]
                        t = cam_params["position_meters"]

                        R_array[frame_num, :, :] = R
                        t_array[frame_num, :] = t

                        frame_num += 1
                        pbar.update(1)

                np.savez(save_path + f"{vid_name}.npz", R=R_array, t=t_array)



