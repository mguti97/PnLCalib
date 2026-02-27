import os
import sys
import json
import glob
import time
import yaml
import torch
import zipfile
import argparse
import warnings
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_keypoints import KeypointsDB
from utils.utils_lines import LineKeypointsDB
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    coords_to_dict, complete_keypoints
from utils.utils_calib import FramebyFrameCalib

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to the (kp model) configuration file")
    parser.add_argument("--cfg_l", type=str, required=True,
                        help="Path to the (line model) configuration file")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory")
    parser.add_argument("--split", type=str, required=True,
                        help="Dataset split")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Saving file path")
    parser.add_argument("--weights_kp", type=str, required=True,
                        help="Model (keypoints) weigths to use")
    parser.add_argument("--weights_line", type=str, required=True,
                        help="Model (lines) weigths to use")
    parser.add_argument("--cuda", type=str, default="cuda:0",
                        help="CUDA device index (default: 'cuda:0')")
    parser.add_argument("--kp_th", type=float, default="0.1")
    parser.add_argument("--line_th", type=float, default="0.1")
    parser.add_argument("--max_reproj_err", type=float, default="50")
    parser.add_argument("--main_cam_only", action='store_true')
    parser.add_argument('--use_gt', action='store_true', help='Use ground truth annotations (default: False)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    files = glob.glob(os.path.join(args.root_dir + args.split, "*.jpg"))

    if args.main_cam_only:
        cam_info = json.load(open(args.root_dir + args.split + '/match_info_cam_gt.json'))
        files = [file for file in files if file.split('/')[-1] in cam_info.keys()]
        files = [file for file in files if cam_info[file.split('/')[-1]]['camera'] == 'Main camera center']
        # files = [file for file in files if int(match_info[file.split('/')[-1]]['ms_time']) == \
        #                                             int(match_info[file.split('/')[-1]]['replay_time'])]

    if args.main_cam_only:
        zip_name = args.save_dir + args.split + '_main.zip'
    else:
        zip_name = args.save_dir + args.split + '.zip'

    if args.use_gt:
        if args.main_cam_only:
            zip_name_pred = args.save_dir + args.split + '_main_gt.zip'
        else:
            zip_name_pred = args.save_dir + args.split + '_gt.zip'
    else:
        if args.main_cam_only:
            zip_name_pred = args.save_dir + args.split + '_main_pred.zip'
        else:
            zip_name_pred = args.save_dir + args.split + '_pred.zip'

    print(f"Saving results in {args.save_dir}")
    print(f"file: {zip_name_pred}")

    if args.use_gt:
        transform = T.Resize((540, 960))
        cam = FramebyFrameCalib(960, 540, denormalize=True)

        with zipfile.ZipFile(zip_name_pred, 'w') as zip_file:
            samples, complete = 0., 0.
            for file in tqdm(files, desc="Processing Images"):
                image = Image.open(file)
                file_name = file.split('/')[-1].split('.')[0]
                samples += 1

                json_path = file.split('.')[0] + ".json"
                f = open(json_path)
                data = json.load(f)

                kp_db = KeypointsDB(data, image)
                line_db = LineKeypointsDB(data, image)
                heatmaps, _ = kp_db.get_tensor_w_mask()
                heatmaps = torch.tensor(heatmaps).unsqueeze(0)
                heatmaps_l = line_db.get_tensor()
                heatmaps_l = torch.tensor(heatmaps_l).unsqueeze(0)
                kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
                line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
                kp_dict = coords_to_dict(kp_coords, threshold=0.01)
                lines_dict = coords_to_dict(line_coords, threshold=0.01)

                cam.update(kp_dict, lines_dict)
                final_params_dict = cam.heuristic_voting()
                # final_params_dict = cam.calibrate(5)

                if final_params_dict:
                    complete += 1
                    cam_params = final_params_dict['cam_params']
                    json_data = json.dumps(cam_params)
                    zip_file.writestr(f"camera_{file_name}.json", json_data)

    else:
        device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
        cfg = yaml.safe_load(open(args.cfg, 'r'))
        cfg_l = yaml.safe_load(open(args.cfg_l, 'r'))

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

        transform = T.Resize((540, 960))
        cam = FramebyFrameCalib(960, 540)

        with zipfile.ZipFile(zip_name_pred, 'w') as zip_file:
            samples, complete = 0., 0.
            time0 = time.time()
            for file in tqdm(files, desc="Processing Images"):
                image = Image.open(file)
                file_name = file.split('/')[-1].split('.')[0]
                samples += 1

                with torch.no_grad():
                    image = f.to_tensor(image).float().to(device).unsqueeze(0)
                    image = image if image.size()[-1] == 960 else transform(image)
                    b, c, h, w = image.size()
                    heatmaps = model(image)
                    heatmaps_l = model_l(image)

                    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
                    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
                    kp_dict = coords_to_dict(kp_coords, threshold=args.kp_th)
                    lines_dict = coords_to_dict(line_coords, threshold=args.line_th)
                    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h)

                    cam.update(kp_dict, lines_dict)
                    final_params_dict = cam.heuristic_voting(refine=False, refine_lines=True)

                if final_params_dict:
                    if final_params_dict['rep_err'] <= args.max_reproj_err:
                        complete += 1
                        cam_params = final_params_dict['cam_params']
                        json_data = json.dumps(cam_params)
                        zip_file.writestr(f"camera_{file_name}.json", json_data)
    
    time1 = time.time()

    with zipfile.ZipFile(zip_name, 'w') as zip_file:
        for file in tqdm(files, desc="Processing Images"):
            file_name = file.split('/')[-1].split('.')[0]
            data = json.load(open(file.split('.')[0] + ".json"))
            json_data = json.dumps(data)
            zip_file.writestr(f"{file_name}.json", json_data)

    print(f'Completed {complete} / {samples}')
    print(f'Frame rate {(time1-time0)/len(files)*1000} (ms)')
