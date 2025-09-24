import os

from .model import Model
from .gtools import gazeto3d
import torch
import cv2
import numpy as np


class GazeTR:
    def __init__(self, weights_file):
        self.model = Model().cuda()
        self.model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
        self.model.eval()  # Set the model to evaluation mode

    def get_direction_vector(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img / 255.0, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = {'face': img}
        with torch.no_grad():
            gaze = self.model.forward(img)
        gaze = gaze.detach().cpu().numpy().flatten()
        dVector = gazeto3d(gaze)
        return dVector


if __name__ == '__main__':
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    pretrained_model = os.path.join(parent_dir, "models\\GazeTR-H-ETH.pt")
    GazeTR(pretrained_model)
