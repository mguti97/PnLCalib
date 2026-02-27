import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from utils.utils_keypoints import KeypointsDB
from utils.utils_lines import LineKeypointsDB
from utils.utils_calib import FramebyFrameCalib

sys.path.append("sn_calibration")
sys.path.append("sn_calibration/src")

from evaluate_camera import get_polylines, scale_points, evaluate_camera_prediction
from evaluate_extremities import mirror_labels
from utils.utils_heatmap import complete_keypoints

def correct_labels(data):
    if 'Goal left post left' in data.keys():
        data['Goal left post left '] = copy.deepcopy(data['Goal left post left'])
        del data['Goal left post left']

    return data

def transform_data(image_data, w, h):
    calib_dict = {}
    for line in image_data:
        line_name = line["class"]
        points = line["points"]
        points_list = [{"x": points[i]/w, "y": points[i+1]/h} for i in range(0, len(points), 2)]

        calib_dict[line_name] = points_list

    calib_dict = correct_labels(calib_dict)
    return calib_dict

def remove_complex_entries(data):
    return {
        key: value
        for key, value in data.items()
        if not isinstance(value.get("x"), complex) and not isinstance(value.get("y"), complex)
    }



def get_calibration(image_data, image_path, refine_lines=False):
    image = Image.open(image_path)    
    kp_db = KeypointsDB(image_data, image)
    ln_db = LineKeypointsDB(image_data, image)

    kp_db.get_full_keypoints()
    ln_db.get_lines()

    kp_dict = remove_complex_entries(kp_db.keypoints_final)
    kp_dict = {key: value for key, value in kp_dict.items() if not (45 <= key <= 57)}
    ln_dict = ln_db.lines

    kp_dict, ln_dict = complete_keypoints(kp_dict, ln_dict, w=image.size[0], h=image.size[1])
    
    cam = FramebyFrameCalib(image.size[0], image.size[1])
    cam.update(kp_dict, ln_dict)

    calib = cam.heuristic_voting(refine_lines=refine_lines)

    return calib


def evaluate(gt, prediction, width, height, th=5):
    line_annotations = scale_points(gt, width, height)
    img_groundtruth = line_annotations
    img_prediction = get_polylines(prediction, width, height,
                                   sampling_factor=0.9)

    confusion1, per_class_conf1, reproj_errors1 = evaluate_camera_prediction(img_prediction,
                                                                             img_groundtruth,
                                                                         th)
    confusion2, per_class_conf2, reproj_errors2 = evaluate_camera_prediction(img_prediction,
                                                                             mirror_labels(img_groundtruth),
                                                                         th)

    accuracy1, accuracy2 = 0., 0.
    if confusion1.sum() > 0:
        accuracy1 = confusion1[0, 0] / confusion1.sum()

    if confusion2.sum() > 0:
        accuracy2 = confusion2[0, 0] / confusion2.sum()

    if accuracy1 > accuracy2:
        accuracy = accuracy1
        confusion = confusion1
        per_class_conf = per_class_conf1
        reproj_errors = reproj_errors1
    else:
        accuracy = accuracy2
        confusion = confusion2
        per_class_conf = per_class_conf2
        reproj_errors = reproj_errors2

    return accuracy


root_path = "/mnt/dades/SoccerNet/SoccerNetv3/"
calib_results = pd.DataFrame()
index = 0
for league in os.listdir(root_path):
    if league != "yolo_format":
        league_path = os.path.join(root_path, league)
        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            for match in os.listdir(season_path):
                match_path = os.path.join(season_path, match)
                with open(match_path + "/Labels-v3.json", "r") as file:
                    data = json.load(file)

                list_actions = data["GameMetadata"]["list_actions"]
                list_replays = data["GameMetadata"]["list_replays"]

                for action in list_actions:
                    width = data["actions"][action]["imageMetadata"]["width"]
                    height = data["actions"][action]["imageMetadata"]["height"]
                    image_data = data["actions"][action]["lines"]
                    image_data_format = transform_data(image_data, width, height)
                    image_path = match_path + f"/{action}"
                    calib = get_calibration(image_data_format, image_path, refine_lines=False)
                    if calib:
                        acc_list = []
                        for th in [5, 10, 20]:
                            accuracy = evaluate(image_data_format, calib["cam_params"], width, height, th)
                            acc_list.append(accuracy)
                        results = pd.DataFrame({'league': league, 'season': season, 'match': match, 'image': action, 'width': width, 'height':height, \
                                                'acc@5': acc_list[0], 'acc@10': acc_list[1], 'acc@20': acc_list[2]}, index=[index])
                    else:
                        results = pd.DataFrame({'league': league, 'season': season, 'match': match, 'image': action, \
                                                'acc@5': np.nan, 'acc@10': np.nan, 'acc@20': np.nan}, index=[index])
                    calib_results = pd.concat([calib_results, results])
                    index += 1

                for action in list_replays:
                    width = data["replays"][action]["imageMetadata"]["width"]
                    height = data["replays"][action]["imageMetadata"]["height"]
                    image_data = data["replays"][action]["lines"]
                    image_data_format = transform_data(image_data, width, height)
                    image_path = match_path + f"/{action}"
                    calib = get_calibration(image_data_format, image_path, refine_lines=False)
                    if calib:
                        acc_list = []
                        for th in [5, 10, 20]:
                            accuracy = evaluate(image_data_format, calib["cam_params"], width, height, th)
                            acc_list.append(accuracy)
                        results = pd.DataFrame({'league': league, 'season': season, 'match': match, 'image': action, 'width': width, 'height':height, \
                                                'acc@5': acc_list[0], 'acc@10': acc_list[1], 'acc@20': acc_list[2]}, index=[index])
                    else:
                        results = pd.DataFrame({'league': league, 'season': season, 'match': match, 'image': action, \
                                                'acc@5': np.nan, 'acc@10': np.nan, 'acc@20': np.nan}, index=[index])
                    calib_results = pd.concat([calib_results, results])
                    index += 1


calib_results.to_csv('calibration_results_wo_kp3_w_aux.csv')
