import cv2
from ultralytics import YOLO
import numpy as np
import torch
from copy import deepcopy
from ultralytics.data.augment import LetterBox
from PIL import Image
from ultralytics.utils.plotting import Annotator, colors, save_one_box

model = YOLO('yolov8n-pose.pt')
image = cv2.imread('person.png')

results = model(image)

# model.predict('person.png', save=True, imgsz=320, conf=0.5)

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs


# Show the results

def plot(
        result,
        kpt_radius=5,
        kpt_line=True):
        if img is None and isinstance(result.orig_img, torch.Tensor):
            img = (result.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        annotator = Annotator(
            deepcopy(result.orig_img),
        )

        # Plot Pose results
        if result.keypoints is not None:
            for k in reversed(result.keypoints.data):
                annotator.kpts(k, result.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result()


for r in results:
    im_array = plot(r)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image