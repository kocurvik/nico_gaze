import os
import re

import cv2
from tqdm import tqdm

from .utils import get_undistort_functions, load_calib_data, get_l_r_image_fnames

width = 3840
height = 2160
frame_size = (680, 480)


def getCalibrationPhoto(dirc, chessboard_size):
    cap_L = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap_R = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    num = findMaxNumber(dirc)
    print("Started", num)

    while cap_R.isOpened():
        success_L, image_L = cap_L.read()
        success_R, image_R = cap_R.read()
        if not success_L or not success_R:
            print("Failed to capture images from one or both cameras")
            continue  # Skip the rest of the loop if image capture failed

        grayL = cv2.cvtColor(image_L, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(image_R, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)
        imgL = image_L.copy()
        imgR = image_R.copy()

        if retL and retR == True:
            imgL = cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
            imgR = cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)

        resized_frame_L = cv2.resize(imgL, frame_size)
        resized_frame_R = cv2.resize(imgR, frame_size)
        img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('Img', img_concat_h)

        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(dirc + "left/" + str(num) + "_l.png", image_L)
            cv2.imwrite(dirc + "right/" + str(num) + "_r.png", image_R)
            cv2.imwrite(dirc + "" + str(num) + "_l.png", image_L)
            cv2.imwrite(dirc + "" + str(num) + "_r.png", image_R)
            print("{} saved".format(num))
            num += 1

    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()


def findMaxNumber(directory):
    max_number = -1
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    pattern = r'\d+'

    for filename in files:
        numbers = re.findall(pattern, filename)
        if numbers:  # Check if any numbers were found
            current_max = max(map(int, numbers))
            if current_max > max_number:
                max_number = current_max

    return max_number + 1


def show_real_time(calib_dict, dirc, calib_dir):
    width = 3840
    height = 2160

    cap_L = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap_R = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    num = findMaxNumber(dirc)
    print("Max number: ", num)
    undistort_l, undistort_r = get_undistort_functions(calib_dict, correct_horizon=False)

    while cap_L.isOpened():
        succes, image_L = cap_L.read()
        succes, image_R = cap_R.read()

        img_l = image_L.copy()
        img_r = image_R.copy()

        img_l = undistort_l(img_l)
        img_r = undistort_r(img_r)

        resized_frame_L = cv2.resize(img_l, frame_size)
        resized_frame_R = cv2.resize(img_r, frame_size)

        img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('Unidistored', img_concat_h)
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(dirc + "" + str(num) + "_l.png", image_L)
            cv2.imwrite(dirc + "" + str(num) + "_r.png", image_R)
            print("{} saved".format(num))
            num += 1


def show_undistored(img_folder, calib_dict):
    images_l, images_r = get_l_r_image_fnames(img_folder, 5)
    undistort_l, undistort_r = get_undistort_functions(calib_dict, get_wide=True)

    for fname_l, fname_r in tqdm(zip(images_l, images_r), total=len(images_l)):
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)

        img_l = undistort_l(img_l)
        img_r = undistort_r(img_r)

        resized_frame_L = cv2.resize(img_l, (640, 480))
        resized_frame_R = cv2.resize(img_r, (640, 480))

        img_concat_h = cv2.hconcat([resized_frame_L, resized_frame_R])
        cv2.imshow('unidistored', img_concat_h)
        cv2.waitKey(0)


if __name__ == '__main__':
    par_dir = "C:/Users/Matej/Desktop/bakalarkaGit/bakalarka/BachelorThesis/nico_images/dataset_03/"
    chessboard_size = (7, 4)

    getCalibrationPhoto(par_dir + "calibration/second/", chessboard_size)

    eyetracker = par_dir + "eyetracker/"
    out_dir = par_dir + '/out'
    calib_dir = load_calib_data(out_dir + "/calib_data.npy")

    show_undistored(par_dir + "calibration/", calib_dir)
    show_real_time(calib_dir, eyetracker + "Matej/", calib_dir)
