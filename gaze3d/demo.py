# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import warnings
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch.utils.data import DataLoader

from eval_utils.visualization import get_corrected_gaze
from gaze3d.src.models.gat_model import GaT, HeadDict, MLPHead, Swin3D
from gaze3d.utils_demo import DemoImageData

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# =============================== GLOBALS =============================== #
CMAP = colormaps.get_cmap("brg")
COLOR_NAMES = [
    "mediumvioletred",
    "green",
    "dodgerblue",
    "crimson",
    "goldenrod",
    "DarkSlateGray",
    "saddlebrown",
    "purple",
    "teal",
]
COLORS = [
    (199, 21, 133),
    (0, 128, 0),
    (30, 144, 255),
    (220, 20, 60),
    (218, 165, 32),
    (47, 79, 79),
    (139, 69, 19),
    (128, 0, 128),
    (0, 128, 128),
]
DET_THR = 0.4  # head detection threshold

# ========================= UTILITY FUNCTIONS =========================== #

def load_head_detection_model(device):
    # Load and return the pre-trained head detection model
    ckpt_path = "gaze3d/checkpoints/crowdhuman_yolov5m.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, verbose=False)
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = [1]  # filter by class, i.e. = [1] for heads
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    model = model.to(device)
    model.eval()
    return model


def detect_heads(image, model):
    """
    Detect heads in the image using the provided model.
    Returns a numpy array containing the detected head bboxes and their confidence scores.
    """
    detections = (
        model(image, size=640).pred[0].cpu().numpy()[:, :-1]
    )  # filter out the class column
    return detections


def load_gaze_model(ckpt_path, device):
    # Load and return the pre-trained Gaze-At-Target model
    # Load checkpoint
    model = GaT(
        encoder=Swin3D(pretrained=False),
        head_dict=HeadDict(
            names=["gaze"],
            modules=[
                partial(
                    MLPHead,
                    hidden_dim=256,
                    num_layers=1,
                    out_features=3,
                )
            ],
        ),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def draw_arrow2D(
    image,
    gaze,
    position=None,
    head_size=None,
    d=0.1,
    color=(255, 0, 0),
    thickness=10,
):
    w, h = image.shape[1], image.shape[0]

    if position is None:
        position = [w // 2, h // 2]

    if head_size:
        length = head_size
    else:
        length = w * d

    gaze_dir = gaze / np.linalg.norm(gaze)
    dx = -length * gaze_dir[0]
    dy = -length * gaze_dir[1]

    cv2.arrowedLine(
        image,
        tuple(np.round(position).astype(np.int32)),
        tuple(np.round([position[0] + dx, position[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    return image


def draw_gaze(
    image,
    head_bbox,
    head_pid,
    gaze,
    cmap,
    colors,
    thickness=10,
    thickness_gaze=10,
    fs=0.8,
):
    img_h, img_w = image.shape[0], image.shape[1]
    scale = max(img_h, img_w) / 1920
    fs *= scale
    thickness = int(scale * thickness)
    thickness_gaze = int(scale * thickness_gaze)

    # =============== Draw Prediction =============== #

    xmin, ymin, xmax, ymax = head_bbox[0], head_bbox[1], head_bbox[2], head_bbox[3]
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    # Compute Head Center
    head_center = np.array([(xmin + xmax) // 2, (ymin + ymax) // 2])
    head_radius = max(xmax - xmin, ymax - ymin) // 2
    head_radius = int(head_radius * 1.2)  # enlarge the head circle
    color = colors[head_pid % len(colors)]
    cv2.circle(image, head_center, head_radius + 1, color, thickness)  # head circle

    # Draw header
    header_text = f"P{int(head_pid)}"
    (w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    header_ul = (
        int(head_center[0] - w_text / 2),
        int(head_center[1] - head_radius - 1 - thickness / 2),
    )
    header_br = (
        int(head_center[0] + w_text / 2),
        int(head_center[1] - head_radius - 1 + h_text + 5),
    )
    cv2.rectangle(image, header_ul, header_br, color, -1)  # header bbox
    cv2.putText(
        image,
        header_text,
        (header_ul[0], int(head_center[1] - head_radius - 1 + h_text)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )  # header text

    # Draw 3D gaze vector
    # color of the gaze vector based on the angle between
    # the gaze and the direction of the camera
    gaze = gaze / np.linalg.norm(gaze)
    gaze = torch.tensor(gaze)[None]
    target = torch.tensor([[0.0, 0.0, -1.0]])
    sim = F.cosine_similarity(gaze, target, dim=1, eps=1e-10)
    sim = F.hardtanh_(sim, min_val=-1.0, max_val=1.0)
    angle_gaze = torch.acos(sim)[0] * 180 / np.pi
    angle_gaze /= 180
    color = np.array(cmap(angle_gaze)[:3]) * 255
    image = draw_arrow2D(
        image=image,
        gaze=gaze[0],
        position=head_center,
        head_size=head_radius,
        d=0.05,
        color=color,
        thickness=thickness_gaze,
    )

    return image


# ========================= Main Class =========================== #


class Gaze3DDemo:
    """
    Gaze3DDemo class to predict gaze on videos or images.
    It first detects and tracks heads in the input video/image,
    then predicts gaze using the pre-trained model.
    Finally, it draws the predicted gaze on the input video/image and saves the output.

    It saves the detected/tracks heads and predicted gaze in the output dir as csv files.
    It also saves the output video/image with the predicted gaze drawn in the output dir.

    Args:
        input_filename (str): Name of the clip/image file to process (with extension).
        output_dir (str): Name of the folder where to save the output.
        ckpt_path (str): Path to the pre-trained model checkpoint.
        inference_modality (str): The modality inference to use could be image for video. (options: "image", "video")
        window_stride (int): only for video inference Stride used to create a clip. 1 seems to be provide more stable gaze.
        device (str): Device to use for inference. (options: "cpu", "cuda")
        batch_size (int): Batch size that can fit in RAM or GPU. (consider increasing for faster inference on video)
        num_workers (int): Number of worker for multiprocessing dataloader. (consider increasing for faster inference on video)

    Example:
        demo = Gaze3DDemo(
            input_filename="data/video.mp4",
            output_dir="output",
            ckpt_path="./checkpoints/gat_stwsge_gaze360_gf.ckpt",
            inference_modality="video",
            window_stride=1,
            device="cuda",
            batch_size=16,
            num_workers=4
        )
        demo.run() # detect, track, predict gaze and draw the output

        # optional
        demo.detect_and_track_heads() # detect and track heads
        demo.predict_gaze() # predict gaze
        demo.draw_prediction() # draw the predicted gaze on the input video/image

    """

    def __init__(
        self,
        ckpt_path: str,
        window_stride: int = 1,
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 1,
    ):
        # Initialize arguments
        self.ckpt_path = ckpt_path
        self.window_stride = window_stride
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize models
        self.model = load_gaze_model(self.ckpt_path, self.device)
        self.head_detector = load_head_detection_model(self.device)

        # Initialize data structures
        self.detected_heads = []

    def __detect_and_track_heads(self, image, frame_id):
        # 1. Convert image
        image_np = np.array(image)
        raw_detections = detect_heads(image_np, self.head_detector)
        detections = []
        for k, raw_detection in enumerate(raw_detections):
            bbox, conf = raw_detection[:4], raw_detection[4]
            if conf > DET_THR:
                cls_ = np.array([0.0])
                detection = np.concatenate([bbox, conf[None], cls_])
                detections.append(detection)
        detections = np.stack(detections)

        # # 2. Detect & track head bboxes
        # tracks = self.tracker.update(detections, image_np)
        # if len(tracks) == 0:
        #     pass
        # # pids = (tracks[:, 4] - 1).astype(int)
        # # head_bboxes = torch.from_numpy(tracks[:, :4]).float()
        # # add the frame_id to the tracks
        # tracks = tracks[:, :5]
        # tracks = np.hstack([tracks, np.ones((len(tracks), 1)) * frame_id])
        return detections

    def __process_image(self, image):
        return self.__detect_and_track_heads(image, 1)


    def __draw_prediction_image(self, image, heads, gazes):
        # =============== Draw =============== #
        frame_np = image[..., ::-1]  # BGR >> RGB
        image = frame_np.copy().astype(np.uint8)

        for row, gaze_row in zip(heads, gazes):
            head_box = [row[0], row[1], row[2], row[3]]
            head_pid = 0
            gaze = gaze_row
            image = draw_gaze(image,
                head_box,
                head_pid,
                gaze,
                CMAP,
                COLORS,
                thickness=10,
                thickness_gaze=10,
                fs=0.8,
            )

        # Save the image
        cv2.imshow("gaze3d-direct", cv2.resize(image[:, :, ::-1], None, fx=0.25, fy=0.25))
        cv2.waitKey(0)

    def detect_and_track_heads(self, image):
        return self.__process_image(image)

    def predict_gaze(self, heads, image):
        dataset = DemoImageData(heads, image)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
        gaze_stack = []
        # Iterate over the dataset
        for sample in dataloader:
            # Predict
            with torch.no_grad():
                pred = self.model(sample["images"].to(self.device))
                gaze = torch.nn.functional.normalize(pred["gaze"], p=2, dim=2, eps=1e-8)
                t = gaze.size(1)
                t = ((t // 2) if t == 1 or t % 2 != 0 else (t // 2) - 1)
                gaze_stack.append(gaze[:, t, :].cpu().numpy())
        # Save the gaze predictions
        gaze_stack = np.vstack(gaze_stack)

        return gaze_stack

    def out_gaze_in_3D(self, image_l, image_r, P_l, P_r, debug=False):
        image_l_pil = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
        image_l_pil = Image.fromarray(image_l_pil)
        
        image_r_pil = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
        image_r_pil = Image.fromarray(image_r_pil)

        heads_l = self.detect_and_track_heads(image_l_pil)
        heads_r = self.detect_and_track_heads(image_r_pil)
        
        head_center_l = np.column_stack([heads_l[:, 0] + heads_l[:, 2], heads_l[:, 1] + heads_l[:, 3]]) / 2
        head_center_r = np.column_stack([heads_r[:, 0] + heads_r[:, 2], heads_r[:, 1] + heads_r[:, 3]]) / 2

        head_point_3d = cv2.triangulatePoints(P_l, P_r, head_center_l.T, head_center_r.T)
        head_point_3d = head_point_3d[:3, :] / head_point_3d[3, :]
        
        gaze = self.predict_gaze(heads_l, image_l_pil)
        gaze[:, :2] *= -1

        gaze_3d = get_corrected_gaze(gaze[0], head_point_3d[:, 0])

        if debug:
            self.__draw_prediction_image(image_l, heads_l, gaze)


        return head_point_3d[:, 0], gaze_3d
