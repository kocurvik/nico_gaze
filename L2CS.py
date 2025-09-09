from l2cs import Pipeline, render, gazeto3d
import torch
import numpy as np
import cv2


# https://github.com/ahmednull/l2cs-net

class L2CS:
    def __init__(self, weights_file, arch="ResNet50"):
        self.gaze_pipeline = Pipeline(
            weights=weights_file,
            arch=arch,
            device=torch.device('cpu'),
            confidence_threshold=0.95
        )

    def detect_gaze(self, frame, show=False):
        try:
            results = self.gaze_pipeline.step(frame)
            if show:
                frame = render(frame, results)
            return True, frame, results
        except ValueError:
            print("No eye detected")
            return False, frame, None

    def _getEyeCoordinates(self, img, show=False):
        ret, img, res = self.detect_gaze(img, show=show)
        if ret == False:
            return False, None, None, None
        a, b = int(res.bboxes[0][0]), int(res.bboxes[0][1])
        if (a < 0): a = 0
        if (b < 0): b = 0
        c, d = self._getBoxSize(res.bboxes[0])
        x, y = (int(a + c / 2.0), int(b + d / 2.0))
        return True, x, y, res, (a, b, c, d)

    def compute_head_coordinates(self, img_l, img_r, P_l, P_r, show=False):
        ret_l, x_l, y_l, res_l, box_coor_l = self._getEyeCoordinates(img_l, show=show)
        ret_r, x_r, y_r, res_r, box_coor_r = self._getEyeCoordinates(img_r, show=show)

        point_l = np.array([[x_l], [y_l]], dtype=float)
        point_r = np.array([[x_r], [y_r]], dtype=float)

        point = cv2.triangulatePoints(P_l, P_r, point_l, point_r)
        head_point = (point[:3] / point[3]).reshape(-1)

        return head_point, res_l, box_coor_l

    def _getBoxSize(self, bbox):
        x_min = int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min = int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        return bbox_width, bbox_height

    def get_direction_vector(self, results):
        gaze = np.array([results.pitch[0], results.yaw[0]])
        dVector = gazeto3d(gaze)
        return dVector
