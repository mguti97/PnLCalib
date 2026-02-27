import sys
import cv2
import copy
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from scipy.optimize import least_squares

from utils.utils_optimize import (vector_seq_to_mtx, point_to_line_distance, get_opt_seq_vector, from_vector,
                                  line_plane_intersection, plane_from_P, compute_semantic_optical_flow)


keypoint_world_coords_2D = [[0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
                            [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
                            [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
                            [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
                            [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
                            [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
                            [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
                            [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
                            [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
                            [88.5, 34.], [94., 34.]]  # 57

keypoint_aux_world_coords_2D = [[5.5, 0], [16.5, 0], [88.5, 0], [99.5, 0], [5.5, 13.84], [99.5, 13.84], [16.5, 24.84],
                                [88.5, 24.84], [16.5, 43.16], [88.5, 43.16], [5.5, 54.16], [99.5, 54.16], [5.5, 68],
                                [16.5, 68], [88.5, 68], [99.5, 68]]

line_world_coords_3D = [[[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]], [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]],
                [[0., 30.34, 0.], [0., 30.34, -2.44]], [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]], [[0., 68., 0.], [105., 68., 0.]], [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]], [[0., 0., 0.], [105., 0., 0.]], [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]], [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]], [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]]

keypoint_world_coords_2D = [[x - 52.5, y - 34] for x, y in keypoint_world_coords_2D]
keypoint_aux_world_coords_2D = [[x - 52.5, y - 34] for x, y in keypoint_aux_world_coords_2D]
line_world_coords_3D = [[[x1 - 52.5, y1 - 34, z1], [x2 - 52.5, y2 - 34, z2]] for [[x1, y1, z1], [x2,y2,z2]] in line_world_coords_3D]



def project(frame, R, t, K, k1):
    It = np.eye(4)[:-1]
    It[:, -1] = -t
    P = K @ (R @ It)
    plane_normal, plane_point = plane_from_P(P, t, K[:-1, -1])
    for line in line_world_coords_3D:
        w1 = line[0]
        w2 = line[1]
        p = line_plane_intersection(w1, w2, plane_normal, plane_point)
        if len(p) > 0:
            w1, w2 = p
            i1, _ = cv2.projectPoints(np.array(w1), R, -R @ t, K, np.array([k1, 0., 0., 0.]))
            i2, _ = cv2.projectPoints(np.array(w2), R, -R @ t, K, np.array([k1, 0., 0., 0.]))
            i1 = i1[0,0]
            i2 = i2[0,0]
            if not (int(i1[0]) > 1e5 or int(i1[1]) > 1e5 or int(i2[0]) > 1e5 or int(i2[1]) > 1e5):
                frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 0, 0), 3)

    r = 9.15
    pts1, pts2, pts3 = [], [], []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos, _ = cv2.projectPoints(pos[:3], R, -R @ t, K, np.array([k1, 0., 0., 0.]))
        ipos = ipos[0,0]
        if not (int(ipos[0]) > 1e5 or int(ipos[1]) > 1e5):
            pts1.append([ipos[0], ipos[1]])

    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos, _ = cv2.projectPoints(pos[:3], R, -R @ t, K, np.array([k1, 0., 0., 0.]))
        ipos = ipos[0, 0]
        if not (int(ipos[0]) > 1e5 or int(ipos[1]) > 1e5):
            pts2.append([ipos[0], ipos[1]])

    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos, _ = cv2.projectPoints(pos[:3], R, -R @ t, K, np.array([k1, 0., 0., 0.]))
        ipos = ipos[0, 0]
        if not (int(ipos[0]) > 1e5 or int(ipos[1]) > 1e5):
            pts3.append([ipos[0], ipos[1]])

    XEllipse1 = np.array(pts1, np.int32)
    XEllipse2 = np.array(pts2, np.int32)
    XEllipse3 = np.array(pts3, np.int32)
    frame = cv2.polylines(frame, [XEllipse1], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse2], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse3], False, (255, 0, 0), 3)

    return frame

def rotation_matrix_to_pan_tilt_roll(rotation):
    """
    Decomposes the rotation matrix into pan, tilt and roll angles. There are two solutions, but as we know that cameramen
    try to minimize roll, we take the solution with the smallest roll.
    :param rotation: rotation matrix
    :return: pan, tilt and roll in radians
    """
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = - first_tilt

    sign_first_tilt = 1. if np.sin(first_tilt) > 0. else -1.
    sign_second_tilt = 1. if np.sin(second_tilt) > 0. else -1.

    first_pan = np.arctan2(sign_first_tilt * orientation[0, 2], sign_first_tilt * - orientation[1, 2])
    second_pan = np.arctan2(sign_second_tilt * orientation[0, 2], sign_second_tilt * - orientation[1, 2])
    first_roll = np.arctan2(sign_first_tilt * orientation[2, 0], sign_first_tilt * orientation[2, 1])
    second_roll = np.arctan2(sign_second_tilt * orientation[2, 0], sign_second_tilt * orientation[2, 1])

    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Conversion from euler angles to orientation matrix.
    :param pan:
    :param tilt:
    :param roll:
    :return: orientation matrix
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]])
    rotMat = np.dot(Rpan, np.dot(Rtilt, Rroll))
    return rotMat


class SequentialCalib:
    def __init__(self, iwidth=960, iheight=540, denormalize=False, use_prev=True):
        self.image_width = iwidth
        self.image_height = iheight
        self.denormalize = denormalize
        self.use_prev = use_prev
        self.calibration = None
        self.principal_point = np.array([iwidth/2, iheight/2])
        self.position = None
        self.rotation = None
        self.distortion = None

        self.prev_frame = None
        self.prev_bboxes = None
        self.prev_calibration = None
        self.prev_position = None
        self.prev_rotation = None
        self.prev_distortion = None

        self.seq_num = -1


    def update(self, frame, bboxes, kp_dict, lines_dict):
        self.seq_num += 1
        self.frame = frame
        self.bboxes = bboxes
        self.keypoints_dict = kp_dict
        self.lines_dict = lines_dict

        if self.denormalize:
            self.denormalize_keypoints()

        self.subsets = self.get_keypoints_subsets()

    def denormalize_keypoints(self):
        for kp in self.keypoints_dict.keys():
            self.keypoints_dict[kp]['x'] *= self.image_width
            self.keypoints_dict[kp]['y'] *= self.image_height
        for line in self.lines_dict.keys():
            self.lines_dict[line]['x_1'] *= self.image_width
            self.lines_dict[line]['y_1'] *= self.image_height
            self.lines_dict[line]['x_2'] *= self.image_width
            self.lines_dict[line]['y_2'] *= self.image_height

    def get_keypoints_subsets(self):
        full, main, ground_plane = {}, {}, {}

        for kp in self.keypoints_dict.keys():
            wp = keypoint_world_coords_2D[kp - 1] if kp <= 57 else keypoint_aux_world_coords_2D[kp - 1 - 57]

            full[kp] = {'xi': self.keypoints_dict[kp]['x'], 'yi': self.keypoints_dict[kp]['y'],
                        'xw': wp[0], 'yw': wp[1], 'zw': -2.44 if kp in [12, 15, 16, 19] else 0.}
            if kp <= 30:
                main[kp] = {'xi': self.keypoints_dict[kp]['x'], 'yi': self.keypoints_dict[kp]['y'],
                                      'xw': wp[0], 'yw': wp[1], 'zw': -2.44 if kp in [12, 15, 16, 19] else 0.}
            if kp not in [12, 15, 16, 19]:
                ground_plane[kp] = {'xi': self.keypoints_dict[kp]['x'], 'yi': self.keypoints_dict[kp]['y'],
                                      'xw': wp[0], 'yw': wp[1], 'zw': -2.44 if kp in [12, 15, 16, 19] else 0.}

        return {'full': full, 'main': main, 'ground_plane': ground_plane}

    def get_per_plane_correspondences(self, mode, use_ransac):
        self.obj_pts, self.img_pts, self.ord_pts = None, None, None

        if mode not in ['full', 'main', 'ground_plane']:
            sys.exit("Wrong mode. Select mode between 'full', 'main_keypoints', 'ground_plane'")

        world_points_p1, world_points_p2, world_points_p3 = [], [], []
        img_points_p1, img_points_p2, img_points_p3 = [], [], []
        keys_p1, keys_p2, keys_p3 = [], [], []

        keypoints = self.subsets[mode]
        for kp in keypoints.keys():
            if kp in [12, 16]:
                keys_p2.append(kp)
                world_points_p2.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p2.append([keypoints[kp]['xi'], keypoints[kp]['yi']])

            elif kp in [1, 4, 8, 13, 17, 20, 24, 28]:
                keys_p1.append(kp)
                keys_p2.append(kp)
                world_points_p1.append([keypoints[kp]['xw'], keypoints[kp]['yw'], keypoints[kp]['zw']])
                world_points_p2.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p1.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
                img_points_p2.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
            elif kp in [3, 7, 11, 14, 18, 23, 27, 30]:
                keys_p1.append(kp)
                keys_p3.append(kp)
                world_points_p1.append([keypoints[kp]['xw'], keypoints[kp]['yw'], keypoints[kp]['zw']])
                world_points_p3.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p1.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
                img_points_p3.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
            elif kp in [15, 19]:
                keys_p3.append(kp)
                world_points_p3.append([-keypoints[kp]['zw'], keypoints[kp]['yw'], 0.])
                img_points_p3.append([keypoints[kp]['xi'], keypoints[kp]['yi']])
            else:
                keys_p1.append(kp)
                world_points_p1.append([keypoints[kp]['xw'], keypoints[kp]['yw'], keypoints[kp]['zw']])
                img_points_p1.append([keypoints[kp]['xi'], keypoints[kp]['yi']])

        obj_points, img_points, key_points, ord_points = [], [], [], []

        if mode == 'ground_plane':
            obj_list = [world_points_p1]
            img_list = [img_points_p1]
            key_list = [keys_p1]
        else:
            obj_list = [world_points_p1, world_points_p2, world_points_p3]
            img_list = [img_points_p1, img_points_p2, img_points_p3]
            key_list = [keys_p1, keys_p2, keys_p3]

        if use_ransac > 0.:
            for i in range(len(obj_list)):
                if len(obj_list[i]) >= 4 and not all(item[0] == obj_list[i][0][0] for item in obj_list[i]) \
                        and not all(item[1] == obj_list[i][0][1] for item in obj_list[i]):
                    if i == 0:
                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC, use_ransac)
                        obj_list[i] = [obj for count, obj in enumerate(obj_list[i]) if status[count]==1]
                        img_list[i] = [obj for count, obj in enumerate(img_list[i]) if status[count]==1]
                        key_list[i] = [obj for count, obj in enumerate(key_list[i]) if status[count]==1]

        for i in range(len(obj_list)):
            if len(obj_list[i]) >= 4 and not all(item[0] == obj_list[i][0][0] for item in obj_list[i])\
                    and not all(item[1] == obj_list[i][0][1] for item in obj_list[i]):
                obj_points.append(np.array(obj_list[i], dtype=np.float32))
                img_points.append(np.array(img_list[i], dtype=np.float32))
                key_points.append(key_list[i])
                ord_points.append(i)

        self.obj_pts = obj_points
        self.img_pts = img_points
        self.key_pts = key_points
        self.ord_pts = ord_points

    def get_correspondences(self, mode):
        obj_pts, img_pts, prob_pts = [], [], []
        keypoints = list(set(list(itertools.chain(*self.key_pts))))
        for kp in keypoints:
            obj_pts.append([self.subsets[mode][kp]['xw'], self.subsets[mode][kp]['yw'], self.subsets[mode][kp]['zw']])
            img_pts.append([self.subsets[mode][kp]['xi'], self.subsets[mode][kp]['yi']])

        return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)

    def change_plane_coords(self, w=105, h=68):
        R = np.array([[0,0,-1], [0,1,0], [1,0,0]])
        self.rotation = self.rotation @ R
        if self.ord_pts[0] == 1:
            self.position = np.linalg.inv(R) @ self.position + np.array([-w/2, 0, 0])
        elif self.ord_pts[0] == 2:
            self.position = np.linalg.inv(R) @ self.position + np.array([w/2, 0, 0])

    def reproj_err(self, obj_pts, img_pts):
        if self.calibration is not None:
            err, n = 0, 0
            for i in range(len(obj_pts)):
                proj_point, _ = cv2.projectPoints(obj_pts[i], self.rotation, -self.rotation@self.position,
                                                  self.calibration, self.distortion)
                err_point = (img_pts[i] - proj_point[0, 0])
                err += np.sum(err_point**2)
                n += 1

            return np.sqrt(err/n)
        else:
            return None

    def vector_to_params(self, vector):
        x_focal_length = vector[0]
        y_focal_length = vector[1]
        k1 = vector[2]
        position_meters = vector[3:6]

        rot_vector = np.array(vector[6:])
        rotation, _ = cv2.Rodrigues(rot_vector)

        self.calibration = np.array([[x_focal_length, 0, self.principal_point[0]],
                                     [0, y_focal_length, self.principal_point[1]],
                                     [0, 0, 1]])
        self.position = position_meters
        self.rotation = rotation
        self.distortion = np.array([k1, 0., 0., 0.])

    def projection_from_cam(self, prev=False):
        It = np.eye(4)[:-1]
        if prev:
            It[:, -1] = -self.prev_position
            P = self.prev_calibration @ (self.prev_rotation @ It)
        else:
            It[:, -1] = -self.position
            P = self.calibration @ (self.rotation @ It)
        return P

    def lines_consensus(self, threshold=200):
        P = self.projection_from_cam()
        plane_normal, plane_point = plane_from_P(P, self.position, self.principal_point)

        self.lines_dict_cons = {}
        if plane_normal is not None:
            for key, value in self.lines_dict.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)
                if len(p) == 2:
                    proj1, _ = cv2.projectPoints(p[0], self.rotation, -self.rotation @ self.position,
                                      self.calibration, self.distortion)
                    proj2, _ = cv2.projectPoints(p[1], self.rotation, -self.rotation @ self.position,
                                      self.calibration, self.distortion)
                else:
                    proj1, _ = cv2.projectPoints(np.array(wp1), self.rotation, -self.rotation @ self.position,
                                      self.calibration, self.distortion)
                    proj2, _ = cv2.projectPoints(np.array(wp2), self.rotation, -self.rotation @ self.position,
                                      self.calibration, self.distortion)

                distance1 = point_to_line_distance(proj1[0, 0], proj2[0, 0], np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1[0, 0], proj2[0, 0], np.array([x2, y2]))

                if distance2 <= threshold and distance1 <= threshold:
                    self.lines_dict_cons[key] = value


    def optimizer_seq(self, vector, img_pts, obj_pts, of_pts, of_X):
        P = vector_seq_to_mtx(vector, self.principal_point)
        R, t, Q, k1 = from_vector(vector, self.principal_point)

        if not any(np.isnan(P.flatten())):
            plane_normal, plane_point = plane_from_P(P, self.position, self.principal_point)

            points, proj_points = [], []
            for i in range(len(img_pts)):
                points.append(img_pts[i])
                proj_point, _ = cv2.projectPoints(obj_pts[i], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                proj_points.append(proj_point[0, 0])

            err1 = (np.array(points) - np.array(proj_points)).ravel()

            err2 = []
            for key, value in self.lines_dict_cons.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)

                if len(p) == 2:
                    proj1, _ = cv2.projectPoints(p[0], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                    proj2, _ = cv2.projectPoints(p[1], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                else:
                    proj1, _ = cv2.projectPoints(np.array(wp1), R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                    proj2, _ = cv2.projectPoints(np.array(wp2), R, -R @ t, Q, np.array([k1, 0., 0., 0.]))

                distance1 = point_to_line_distance(proj1[0,0], proj2[0,0], np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1[0,0], proj2[0,0], np.array([x2, y2]))
                err2.append([distance1, distance2])

            if len(of_pts):
                proj_of_pts = []
                for i in range(len(of_pts)):
                    proj_x, _ = cv2.projectPoints(of_X[i], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                    proj_of_pts.append(proj_x[0,0])

                err_of = (of_pts - np.array(proj_of_pts)).ravel()
                return np.concatenate((err_of, err1, np.array(err2).ravel()))
            else:
                return np.concatenate((err1, np.array(err2).ravel()))

        else:
            err = []
            for i in range(len(of_pts)+len(img_pts)+len(self.lines_dict_cons)+len(of_pts)):
                err.append([np.inf, np.inf])
            return np.array(err).ravel()


    def optimizer(self, vector, img_pts, obj_pts):
        P = vector_seq_to_mtx(vector, self.principal_point)
        R, t, Q, k1 = from_vector(vector, self.principal_point)

        if not any(np.isnan(P.flatten())):
            plane_normal, plane_point = plane_from_P(P, self.position, self.principal_point)

            points, proj_points = [], []
            for i in range(len(img_pts)):
                points.append(img_pts[i])
                proj_point, _ = cv2.projectPoints(obj_pts[i], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                proj_points.append(proj_point[0,0])

            err1 = (np.array(points) - np.array(proj_points)).ravel()

            err2 = []
            for key, value in self.lines_dict_cons.items():
                y1, y2 = value['y_1'], value['y_2']
                x1, x2 = value['x_1'], value['x_2']

                wp1, wp2 = line_world_coords_3D[key - 1]
                p = line_plane_intersection(wp1, wp2, plane_normal, plane_point)

                if len(p) == 2:
                    proj1, _ = cv2.projectPoints(p[0], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                    proj2, _ = cv2.projectPoints(p[1], R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                else:
                    proj1, _ = cv2.projectPoints(np.array(wp1), R, -R @ t, Q, np.array([k1, 0., 0., 0.]))
                    proj2, _ = cv2.projectPoints(np.array(wp2), R, -R @ t, Q, np.array([k1, 0., 0., 0.]))


                distance1 = point_to_line_distance(proj1[0,0], proj2[0,0], np.array([x1, y1]))
                distance2 = point_to_line_distance(proj1[0,0], proj2[0,0], np.array([x2, y2]))
                err2.append([distance1, distance2])

            return np.concatenate((err1, np.array(err2).ravel()))

        else:
            err = []
            for i in range(len(img_pts)+len(self.lines_dict_cons)):
                err.append([np.inf, np.inf])
            return np.array(err).ravel()


    def get_cam_params(self, mode='full', use_ransac=0, refine=False):
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        flags = flags | cv2.CALIB_FIX_TANGENT_DIST | \
                cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY
        flags = flags | cv2.CALIB_FIX_K2 | \
                cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | \
                cv2.CALIB_FIX_K6

        self.get_per_plane_correspondences(mode=mode, use_ransac=use_ransac)

        if len(self.obj_pts) == 0:
            return None, None


        obj_pts, img_pts = self.get_correspondences(mode)
        if len(obj_pts) < 6:
            if self.prev_calibration is not None:
                flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_pts,
                self.img_pts,
                (self.image_width, self.image_height),
                self.prev_calibration,
                self.prev_distortion,
                flags=flags,
            )

        else:
            mtx = cv2.initCameraMatrix2D(
                self.obj_pts,
                self.img_pts,
                (self.image_width, self.image_height),
                aspectRatio=1.0,
            )
            if not np.isnan(np.min(mtx)):
                flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    [obj_pts],
                    [img_pts],
                    (self.image_width, self.image_height),
                    mtx,
                    None,
                    flags=flags,
                )
            else:
                ret = False

        if ret:
            self.calibration = mtx
            R, _ = cv2.Rodrigues(rvecs[0])
            self.rotation = R
            self.position = (-np.transpose(self.rotation) @ tvecs[0]).T[0]
            self.distortion = np.array([dist[0,0], 0., 0., 0.])

            if self.ord_pts[0] != 0:
                self.change_plane_coords()

            obj_pts, img_pts = self.get_correspondences(mode)
            rep_err = self.reproj_err(obj_pts, img_pts)

            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40, 10))
            # ax1.imshow(self.frame)
            # for kp in self.subsets[mode]:
            #     ax1.scatter(self.subsets[mode][kp]["xi"], self.subsets[mode][kp]["yi"], s=20, c='r')
            #
            # frame_proj1 = project(copy.deepcopy(self.frame), self.rotation, self.position, self.calibration, self.distortion[0])
            # ax3.imshow(frame_proj1)
            # ax3.set_title(rep_err)

            if refine:
                if not np.isnan(rep_err):
                    self.lines_consensus()
                    # ax2.imshow(self.frame)
                    # for line in self.lines_dict_cons:
                    #     ax2.scatter(self.lines_dict_cons[line]["x_1"], self.lines_dict_cons[line]["y_1"], s=20, c='r')
                    #     ax2.scatter(self.lines_dict_cons[line]["x_2"], self.lines_dict_cons[line]["y_2"], s=20, c='r')


                    vector = get_opt_seq_vector(self.calibration, self.distortion[0], self.position, self.rotation)
                    if self.prev_frame is not None:
                        of_pts, of_X = compute_semantic_optical_flow(self.prev_frame, self.frame,
                                                             self.prev_calibration,
                                                             self.prev_rotation, self.prev_position,
                                                             k1=self.prev_distortion[0],
                                                             bboxes=self.prev_bboxes)

                        res = least_squares(self.optimizer_seq, vector, verbose=0, ftol=1e-4, x_scale="jac",
                                            method='trf',
                                            args=(img_pts, obj_pts, of_pts, of_X))
                    else:
                        res = least_squares(self.optimizer, vector, verbose=0, ftol=1e-4, x_scale="jac",
                                            method='trf',
                                            args=(img_pts, obj_pts))

                    vector_opt = res['x']
                    if not any(np.isnan(vector_opt)):
                        self.vector_to_params(vector_opt)
                        rep_err = self.reproj_err(obj_pts, img_pts)

                        # frame_proj2 = project(copy.deepcopy(self.frame), self.rotation, self.position, self.calibration,
                        #                      self.distortion[0])
                        # ax4.imshow(frame_proj2)
                        # ax4.set_title(rep_err)
                        # fig.savefig(f'debug_img/test_{self.seq_num}_{mode}_{use_ransac}.png', dpi=fig.dpi)

            pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)

            pan = np.rad2deg(pan)
            tilt = np.rad2deg(tilt)
            roll = np.rad2deg(roll)

            cam_params = {"pan_degrees": pan,
                          "tilt_degrees": tilt,
                          "roll_degrees": roll,
                          "x_focal_length": self.calibration[0,0],
                          "y_focal_length": self.calibration[1,1],
                          "principal_point": [self.principal_point[0], self.principal_point[1]],
                          "position_meters": [self.position[0], self.position[1], self.position[2]],
                          "rotation_matrix": [[self.rotation[0, 0], self.rotation[0, 1], self.rotation[0, 2]],
                                              [self.rotation[1, 0], self.rotation[1, 1], self.rotation[1, 2]],
                                              [self.rotation[2, 0], self.rotation[2, 1], self.rotation[2, 2]]],
                          "radial_distortion": [self.distortion[0], 0., 0., 0., 0., 0.],
                          "tangential_distortion": [0., 0.],
                          "thin_prism_distortion": [0., 0., 0., 0.]}

            return cam_params, rep_err
        else:
            return None, None

    def save_to_prev(self, frame, cam_params):
        self.prev_frame = copy.deepcopy(self.frame)
        self.prev_bboxes = copy.deepcopy(self.bboxes)
        self.prev_calibration = np.array([[cam_params["x_focal_length"], 0, cam_params["principal_point"][0]],
                                          [0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
                                          [0, 0, 1]])
        self.prev_rotation = np.array(cam_params["rotation_matrix"])
        self.prev_position = np.array(cam_params["position_meters"])
        self.prev_distortion = np.array(cam_params["radial_distortion"])[:4]


    def reinit_prev(self):
        self.prev_frame = None
        self.prev_bboxes = None
        self.prev_position = None
        self.prev_distortion = None


    def heuristic_voting(self, refine=False, max_reproj_err=5., th=5.):
        final_results = []
        for mode in ['full']:
            for use_ransac in [0, 5, 10]:
                cam_params, ret = self.get_cam_params(mode=mode, use_ransac=use_ransac, refine=refine)
                if ret:
                    result_dict = {'mode': mode, 'use_ransac': use_ransac, 'rep_err': ret,
                                   'cam_params': cam_params, 'calib_plane': self.ord_pts[0]}
                    final_results.append(result_dict)

        if final_results:
            final_results.sort(key=lambda x: (x['rep_err'], x['mode']))
            for res in final_results:
                if res['mode'] == 'full' and res['use_ransac'] == 0 and res['rep_err'] <= th:
                    return res
            if final_results[0]["rep_err"] < max_reproj_err:
                if self.use_prev:
                    self.save_to_prev(self.frame, final_results[0]["cam_params"])
                return final_results[0]
            else:
                self.reinit_prev()
                return None
        else:
            self.reinit_prev()
            return None



