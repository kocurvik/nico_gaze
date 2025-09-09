import argparse
import os

from tqdm import tqdm
import cv2

from utils import get_l_r_image_fnames, load_calib_data, get_undistort_functions, crop_image, save_array, load_data, \
    get_data, convert_mm_to_pixels
import vectors
import numpy as np
import visualization
import plane as planeDetector
from charts import show_charts
from GazeTR_net.GazeTR import GazeTR
from L2CS import L2CS
from error import Error

frameSize = (640, 480)
frameSize = (1280, 720)


def intersection_gaze_plane_main(img_folder, calib_dir, gaze_L2CS, gaze_GazeTR, chessboard_dim=35.0,
                                 chessboard_size=(7, 4), index=1, withGlasses=False, debug=0, cropp=False):
    images_l, images_r = get_l_r_image_fnames(img_folder, datasetFolder=True, withGlasses=withGlasses, index=index,
                                              name='')

    undistort_l, undistort_r = get_undistort_functions(calib_dir, get_wide=False)

    list_errors_L2CS = np.array([])
    list_errors_gazeTR = np.array([])

    # detect plane and chessboards for calculation and visualization
    if (len(images_l) > 0):
        img_l = cv2.imread(images_l[0])
        img_l = undistort_l(img_l)

        plane, chessboard_middle_points = planeDetector.detect_plane(img_l, calib_dir, chessboard_dim, chessboard_size)

    for fname_l, fname_r in tqdm(zip(images_l, images_r), total=len(images_l)):
        # print(fname_l)
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)

        img_l = undistort_l(img_l)
        img_r = undistort_r(img_r)

        error_L2CS, error_gazeTR = intersection_gaze_plane(img_l, img_r, calib_dir, gaze_L2CS, gaze_GazeTR, plane,
                                                           chessboard_middle_points, index, debug=debug,
                                                           path_img=fname_l, cropp=cropp)
        list_errors_L2CS = np.append(list_errors_L2CS, error_L2CS)
        list_errors_gazeTR = np.append(list_errors_gazeTR, error_gazeTR)

    return list_errors_L2CS, list_errors_gazeTR


def intersection_gaze_plane(img_l, img_r, calib_dir, L2CS, GazeTR, plane, chessboard_middle_points, index, debug=0,
                            path_img=None, cropp=False, ):
    P_l = calib_dir['P_l']
    P_r = calib_dir['P_r']

    view_point = chessboard_middle_points[chessboard_middle_points[:, 3] == index][:, :3].flatten()

    head_point, res, head_box = L2CS.compute_head_coordinates(img_l, img_r, P_l, P_r, show=False)
    length = head_box[2]

    img_detect_gaze = img_l.copy()

    if cropp:
        img_detect_gaze = crop_image(img_detect_gaze, head_box)
        ret, img_detect_gaze, res = L2CS.detect_gaze(img_detect_gaze)


    dVector_gazeTR = GazeTR.get_direction_vector(img_detect_gaze.copy())
    dVector_L2CS = L2CS.get_direction_vector(res)

    dVectorCorrect = vectors.get_direction_vector_from_line(head_point, view_point)

    point_on_line_L2CS = vectors.add_dVector_3D(head_point, dVector_L2CS, length/2)
    point_on_line_2D = convert_mm_to_pixels(point_on_line_L2CS, P_l)
    head_point_2D = convert_mm_to_pixels(head_point, P_l)
    view_point_2D = convert_mm_to_pixels(view_point, P_l)

    cv2.arrowedLine(img_l, tuple(head_point_2D),
                    tuple(point_on_line_2D), (0, 255, 0),
                    5, cv2.LINE_AA, tipLength=0.18)

    point_on_line_gazeTR = vectors.add_dVector_3D(head_point, dVector_gazeTR, length / 2)
    point_on_line_2D_gaze = convert_mm_to_pixels(point_on_line_gazeTR, P_l)

    cv2.arrowedLine(img_l, tuple(head_point_2D),
                    tuple(point_on_line_2D_gaze), (128, 0, 128),
                    5, cv2.LINE_AA, tipLength=0.18)

    cv2.arrowedLine(img_l, tuple(head_point_2D),
                    tuple(view_point_2D), (0, 165, 255),
                    5, cv2.LINE_AA, tipLength=0.18)

    intersection_point_L2CS = planeDetector.find_intersection(plane, head_point, dVector_L2CS)
    intersection_point_gazeTR = planeDetector.find_intersection(plane, head_point, dVector_gazeTR)

    error_L2CS = Error(view_point, intersection_point_L2CS, dVectorCorrect, dVector_L2CS)
    error_gazeTR = Error(view_point, intersection_point_gazeTR, dVectorCorrect, dVector_gazeTR)

    if debug > 0:
        # 3D visualization
        fig = visualization.create_base_figure()
        for point in chessboard_middle_points:
            visualization.plot_3D_point(fig, point[:3], name=int(point[3]), color='red')
        visualization.plot_3D_point(fig, head_point, name="Head Point", color="red")
        visualization.plot_line(fig, head_point, intersection_point_L2CS, color="green", name="L2CS")
        visualization.plot_line(fig, head_point, intersection_point_gazeTR, color="purple", name="GazeTR")
        visualization.visualize_plane(fig, plane, np.array([intersection_point_L2CS, intersection_point_gazeTR]),
                                      chessboard_middle_points)

        visualization.plot_line(fig, view_point, head_point, color="orange", name="correct point")

        # visualization.plot_line(fig, view_point, intersection_point_L2CS, color="red", name="error")
        # visualization.plot_line(fig, view_point, intersection_point_gazeTR, color="red", name="error")

        fig.show()
        resized_frame_L = cv2.resize(img_l, frameSize)
        # resized_frame_R = cv2.resize(img_r, frameSize)
        img_concat_h = resized_frame_L
        # img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('intersection', img_concat_h)
        cv2.waitKey(0)
    return error_L2CS, error_gazeTR





def calculate(eyeTrackerDir, save_dir, pretrained_model_L2CS, pretrained_model_GazeTR, calib_file, chessboard_dim, chessboard_size,
              prefix='', cropp=False, debug=0):
    calib_dir = load_calib_data(calib_file)
    if prefix != '': prefix += "_"

    gaze_L2CS = L2CS(pretrained_model_L2CS)
    gaze_GazeTR = GazeTR(pretrained_model_GazeTR)

    folder_L2CS = "/L2CS"
    folder_GazeTR = "/GazeTR"

    path_with_glasses = f"/{prefix}with_glasses.npy"
    path_no_glasses = f"/{prefix}no_glasses.npy"

    num_of_image_folders = 20

    list_errors_with_glasses_L2CS = np.array([])
    list_errors_no_glasses_L2CS = np.array([])

    list_errors_with_glasses_GazeTR = np.array([])
    list_errors_no_glasses_GazeTR = np.array([])

    boolean = [False, True]

    for glasses in boolean:
        for index in range(1, num_of_image_folders + 1):
            errors_L2CS, errors_GazeTR = intersection_gaze_plane_main(eyeTrackerDir, calib_dir, gaze_L2CS,
                                                                         gaze_GazeTR,
                                                                         chessboard_dim, chessboard_size, index=index,
                                                                         withGlasses=glasses, debug=debug, cropp=cropp)
            if glasses:
                list_errors_with_glasses_L2CS = np.append(list_errors_with_glasses_L2CS, errors_L2CS)
                list_errors_with_glasses_GazeTR = np.append(list_errors_with_glasses_GazeTR, errors_GazeTR)
            else:
                list_errors_no_glasses_L2CS = np.append(list_errors_no_glasses_L2CS, errors_L2CS)
                list_errors_no_glasses_GazeTR = np.append(list_errors_no_glasses_GazeTR, errors_GazeTR)

    save_array(list_errors_with_glasses_L2CS, save_dir + folder_L2CS + path_with_glasses)
    save_array(list_errors_no_glasses_L2CS, save_dir + folder_L2CS + path_no_glasses)

    save_array(list_errors_with_glasses_GazeTR, save_dir + folder_GazeTR + path_with_glasses)
    save_array(list_errors_no_glasses_GazeTR, save_dir + folder_GazeTR + path_no_glasses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model_L2CS', type=str, help='path to pre-trained L2CS model')
    parser.add_argument('pretrained_model_GazeTR', type=str, help='path to pre-trained GazeTR model')
    parser.add_argument('eyeTrackerDir', type=str, help='directory where the images for eyetracking are saved')
    parser.add_argument('save_dir', type=str, help='directory where the calculated data like errors gets saved to')
    parser.add_argument('calib_file', type=str, help='file with calibration information')
    parser.add_argument('-d', '--debug', type=int, default=0,
                        help='whether to debug 1 shows detected gaze and visualize scene in 3D')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    chessboard_dim = 42.0
    chessboard_size = (7, 4)
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    pretrained_model_L2CS = os.path.join(parent_dir, "models\\L2CSNet_gaze360.pkl")
    pretrained_model_GazeTR = os.path.join(parent_dir, "models\\GazeTR-H-ETH.pt")
    debug = 0
    eyeTrackerDir = parent_dir + '\\dataset\\eyetracker'
    out_dir = os.path.join(parent_dir, 'out')
    save_dir = out_dir + '\\data'
    calib_file = os.path.join(out_dir, 'calib_data.npy')

    # args = parse_args()
    # pretrained_model_L2CS = args.pretrained_model_L2CS
    # pretrained_model_GazeTR = args.pretrained_model_GazeTR
    # save_dir = args.save_dir
    # eyeTrackerDir = args.eyeTracker_dir
    # calib_file = args.calib_file
    # debug = args.debug

    # # Calculate errors with origin image
    # calculate(eyeTrackerDir, save_dir, pretrained_model_L2CS, pretrained_model_GazeTR, calib_file, chessboard_dim,
    #         chessboard_size, prefix="", cropp=False, debug=debug)

    # # Calculate errors with cropped image
    # calculate(eyeTrackerDir, save_dir, pretrained_model_L2CS, pretrained_model_GazeTR, calib_file, chessboard_dim, chessboard_size,
    #           prefix='cropped', cropp=True, debug=debug)

    # show calculated data
    show_charts(save_dir)


