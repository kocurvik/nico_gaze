import argparse
import json
import os

from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import cv2

from sklearn.linear_model import LinearRegression

from GazeNet.inference import GazeNetInference
from eval_utils.visualization import plot_angle_distribution, yaw_pitch_from_direction, direction_from_yaw_pitch, \
    get_corrected_gaze

from gaze3d.demo import Gaze3DDemo
from eval_utils.image import get_l_r_image_fnames, load_calib_data, get_undistort_functions, crop_image, \
    convert_mm_to_pixels
import numpy as np
from eval_utils import plane as planeDetector
from GazeTR_net.GazeTR import GazeTR

from L2CS.L2CS import L2CS
from eval_utils.error import get_direction_vector_from_line, get_result_dict, vectors_angle

frameSize = (1280, 720)

class Evaluator():
    def __init__(self, dataset_dir, debug=0):
        self.chessboard_dim = 42.0
        self.chessboard_size = (7, 4)

        self.debug = debug

        self.eyeTrackerDir = dataset_dir
        self.save_dir = os.path.join('eval_data')
        self.calib_file = os.path.join('eval_data', 'calib_data.npy')

        self.models = {'L2CS': None, 'GazeTR': None, '3DGazeNet': None, 'gaze3d': None}

    def init_models(self):
        pretrained_model_L2CS = os.path.join('L2CS', 'weights', 'L2CSNet_gaze360.pkl')
        pretrained_model_GazeTR = os.path.join('GazeTR_net', 'weights', 'GazeTR-H-ETH.pt')
        pretrained_model_gaze3d = os.path.join('gaze3d', 'checkpoints', 'gat_stwsge_gaze360_gf.ckpt')

        self.gaze_L2CS = L2CS(pretrained_model_L2CS)
        self.GazeTR = GazeTR(pretrained_model_GazeTR)
        self._3DGazeNet = GazeNetInference(0.5, 224)
        self.Gaze3D = Gaze3DDemo(pretrained_model_gaze3d, device='cuda')
        self.models = {'L2CS': self.gaze_L2CS, 'GazeTR': self.GazeTR, '3DGazeNet':self._3DGazeNet, 'gaze3d': self.Gaze3D}

    def eval(self):
        self.results = []
        self.calib_dir = load_calib_data(self.calib_file)

        num_of_image_folders = 20

        for index in range(1, num_of_image_folders + 1):
            self.eval_dir(self.eyeTrackerDir, index=index)

        with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f)
            print("Saved results")


    def load_errors(self):
        with open(os.path.join(self.save_dir, 'results.json'), 'r') as f:
            self.results = json.load(f)
            print("Loaded results")

    def eval_dir(self, img_folder, index=1):
        images_l, images_r = get_l_r_image_fnames(img_folder, datasetFolder=True, index=index, name='')
        undistort_l, undistort_r = get_undistort_functions(self.calib_dir, get_wide=False)

        # detect plane and chessboards for calculation and visualization
        if (len(images_l) > 0):
            img_l = cv2.imread(images_l[0])
            img_l = undistort_l(img_l)
            plane, chessboard_middle_points = planeDetector.detect_plane(img_l, self.calib_dir, self.chessboard_dim, self.chessboard_size)

        for fname_l, fname_r in tqdm(zip(images_l, images_r), total=len(images_l)):
            # print(fname_l)
            img_l = cv2.imread(fname_l)
            img_r = cv2.imread(fname_r)

            img_l = undistort_l(img_l)
            img_r = undistort_r(img_r)

            fname_l_short = os.path.normpath(fname_l).split(os.path.normpath(img_folder))[1]
            fname_r_short = os.path.normpath(fname_r).split(os.path.normpath(img_folder))[1]

            self.eval_single(img_l, img_r, plane, chessboard_middle_points, index, fname_l_short, fname_r_short)


    def eval_single(self, img_l, img_r, plane, chessboard_middle_points, index, fname_l, fname_r):
        P_l = self.calib_dir['P_l']
        P_r = self.calib_dir['P_r']
        
        head_point = {}
        direction = {}
        direction_gt = {}

        view_point = chessboard_middle_points[chessboard_middle_points[:, 3] == index][:, :3].flatten()

        head_point['L2CS'], res, head_box = self.models['L2CS'].compute_head_coordinates(img_l.copy(), img_r.copy(), P_l, P_r, show=self.debug > 1)
        head_point['GazeTR'] = head_point['L2CS']

        img_detect_gaze = crop_image(img_l.copy(), head_box)
        direction['GazeTR'] = self.models['GazeTR'].get_direction_vector(img_detect_gaze.copy())
        direction['L2CS'] = self.models['L2CS'].get_direction_vector(res, head_point['L2CS'])
        head_point['3DGazeNet'], direction['3DGazeNet'] = self.models['3DGazeNet'].out_gaze_in_3D(img_l.copy(), img_r, P_l, P_r, debug=self.debug > 1)
        head_point['gaze3d'], direction['gaze3d'] = self.models['gaze3d'].out_gaze_in_3D(img_l.copy(), img_r, P_l, P_r, debug=self.debug > 1)

        direction_gt['L2CS'] = get_direction_vector_from_line(head_point['L2CS'], view_point)
        direction_gt['GazeTR'] = direction_gt['L2CS']
        direction_gt['3DGazeNet'] = get_direction_vector_from_line(head_point['3DGazeNet'], view_point)
        direction_gt['gaze3d'] = get_direction_vector_from_line(head_point['gaze3d'], view_point)

        if self.debug:
            for model_name in self.models.keys():
                self.draw_gaze(model_name, img_l.copy(), P_l, head_point[model_name], direction[model_name])


        intersection_point = {k: planeDetector.find_intersection(plane, head_point[k], direction[k]) for k in self.models.keys()}

        for k in self.models.keys():
            d = get_result_dict(view_point, intersection_point[k], direction_gt[k], direction[k], head_point[k])
            d['method'] = k
            d['fname_l'] = fname_l
            d['fname_r'] = fname_r
            d['plane'] = plane.tolist()
            d['index'] = index
            d['participant'] = fname_l.split('\\')[1]
            self.results.append(d)

        # if self.debug > 1:
        #     # 3D visualization
        #     fig = visualization.create_base_figure()
        #     for point in chessboard_middle_points:
        #         visualization.plot_3D_point(fig, point[:3], name=int(point[3]), color='red')
        #     visualization.plot_3D_point(fig, head_point['L2CS'], name="Head Point", color="red")
        #     visualization.plot_line(fig, head_point['L2CS'], intersection_point_L2CS, color="green", name="L2CS")
        #     visualization.plot_line(fig, head_point['L2CS'], intersection_point_gazeTR, color="purple", name="GazeTR")
        #     visualization.visualize_plane(fig, plane, np.array([intersection_point_L2CS, intersection_point_gazeTR]),
        #                                   chessboard_middle_points)
        #
        #     visualization.plot_line(fig, view_point, head_point['L2CS'], color="orange", name="correct point")
        #
        #     # visualization.plot_line(fig, view_point, intersection_point_L2CS, color="red", name="error")
        #     # visualization.plot_line(fig, view_point, intersection_point_gazeTR, color="red", name="error")
        #
        #     fig.show()
        #     resized_frame_L = cv2.resize(img_l, frameSize)
        #     # resized_frame_R = cv2.resize(img_r, frameSize)
        #     img_concat_h = resized_frame_L
        #     # img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        #     cv2.imshow('intersection', img_concat_h)
        #     cv2.waitKey(0)

    def print_tables(self):
        def err_dict():
            return {'with_glasses': {}, 'no_glasses': {}, 'both': {}}
        
        errors = {'angular_error_3d': err_dict(), 'angular_error_2d': err_dict(), 'distance_error_3d': err_dict()}

        for error_type in errors.keys():
            for model_name in self.models.keys():
                errors[error_type][model_name] = [x[error_type] for x in self.results if x['method'] == model_name]            
                
        
        tab_tex = PrettyTable(['Method', 'Mean Angular Error', 'Mean Distance', 'Precision@10cm', 'Precision@20cm', 'Precision@50cm'])
        tab_tex.float_format = '0.2'

        methods = ['GazeTR', 'L2CS', '3DGazeNet', 'gaze3d']

        for model_name in methods:
            mean_angular_3d = np.mean(errors['angular_error_3d'][model_name])

            median_distance = np.median(errors['distance_error_3d'][model_name]) / 10.0
            distance_precision_100 = 100 * np.sum(np.array(errors['distance_error_3d'][model_name]) < 100) / len(
                errors['distance_error_3d'][model_name])
            distance_precision_200 = 100 * np.sum(np.array(errors['distance_error_3d'][model_name]) < 200) / len(
                errors['distance_error_3d'][model_name])
            distance_precision_500 = 100 * np.sum(np.array(errors['distance_error_3d'][model_name]) < 500) / len(
                errors['distance_error_3d'][model_name])

            tab_tex.add_row([model_name, mean_angular_3d, median_distance, distance_precision_100, distance_precision_200, distance_precision_500])

        print(tab_tex)
        print("Latex ***")
        print(tab_tex.get_latex_string())


    def draw_gaze(self, window_name, img, P_l, head_point, dVector):
        head_point_2D = convert_mm_to_pixels(head_point, P_l)
        view_point = head_point + 1000 * dVector
        view_point_2D = convert_mm_to_pixels(view_point, P_l)
        view_point_2D_dir = head_point_2D + (500 * dVector[:2]).astype(int)

        print(window_name, dVector)

        cv2.arrowedLine(img, tuple(head_point_2D), tuple(view_point_2D), (0, 255, 0), 5, cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(img, tuple(head_point_2D), tuple(view_point_2D_dir), (0, 0, 255), 5, cv2.LINE_AA, tipLength=0.18)
        cv2.imshow(window_name, cv2.resize(img, None, fx=0.25, fy=0.25))
        cv2.waitKey(0)

    def plot_distributions(self):
        os.makedirs('figs', exist_ok=True)
        directions = [x['gt_direction'] for x in self.results if x['method'] == 'L2CS']
        plot_angle_distribution('GT', directions)

        for model_name in self.models.keys():
            directions = [x['est_direction'] for x in self.results if x['method'] == model_name]
            plot_angle_distribution(model_name, directions)

    def plot_cummulative_errors(self):
        def err_dict():
            return {'with_glasses': {}, 'no_glasses': {}, 'both': {}}

        errors = {'angular_error_3d': err_dict(), 'angular_error_2d': err_dict(), 'distance_error_3d': err_dict()}

        for error_type in errors.keys():
            for model_name in self.models.keys():
                errors[error_type][model_name] = [x[error_type] for x in self.results if x['method'] == model_name]

        methods = ['GazeTR', 'L2CS', '3DGazeNet', 'gaze3d']

        angles = np.linspace(0, 30, 91)

        plt.figure(figsize=(7, 4))

        for model_name in methods:
            errs = np.array(errors['angular_error_3d'][model_name])
            ys = [100 * np.sum(errs < x) / len(errs) for x in angles]
            plt.plot(angles, ys, label = 'model_name', linewidth=2)

        plt.ylim([0, 100.0])
        plt.xlim([angles[0], angles[-1]])

        large_size = 20
        small_size = 16

        plt.ylabel('Precision (%)', fontsize=large_size)
        plt.xlabel('Angular Error ($^\circ$)', fontsize=large_size)
        plt.tick_params(axis='x', which='major', labelsize=small_size)
        plt.tick_params(axis='y', which='major', labelsize=small_size)

        plt.savefig(f'figs/cum_angle.pdf', bbox_inches='tight', pad_inches=0.1)

        distances = np.linspace(0, 100, 101)

        plt.figure(figsize=(7, 4))

        for model_name in methods:
            errs = np.array(errors['distance_error_3d'][model_name])
            ys = [100 * np.sum(errs < x * 10) / len(errs) for x in distances]
            plt.plot(distances, ys, label = 'model_name', linewidth=2)

        plt.ylim([0, 100.0])
        plt.xlim([distances[0], distances[-1]])

        large_size = 20
        small_size = 16

        plt.ylabel('Precision (%)', fontsize=large_size)
        plt.xlabel('Distance Error (cm)', fontsize=large_size)
        plt.tick_params(axis='x', which='major', labelsize=small_size)
        plt.tick_params(axis='y', which='major', labelsize=small_size)

        plt.savefig(f'figs/cum_dist.pdf', bbox_inches='tight', pad_inches=0.1)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-d', '--debug', type=int, default=0, help='enable debug output')
    parser.add_argument('dataset_dir', type=str, help='directory with calibration images')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args.dataset_dir, debug=args.debug)
    if args.load:
        evaluator.load_errors()
    else:
        evaluator.init_models()
        evaluator.eval()
    evaluator.print_tables()
    evaluator.plot_cummulative_errors()

    evaluator.plot_distributions()




