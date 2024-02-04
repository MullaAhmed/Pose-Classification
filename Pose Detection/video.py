import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from copy import deepcopy
import torch

model = YOLO('yolov8n-pose.pt')


video_path = "input/pose-estimation-synchronised-sample-b.mp4"
cap = cv2.VideoCapture(0)

def plot(
        result,
        kpt_radius=5,
        kpt_line=True,
        img=None,):
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

while cap.isOpened():
  
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame =plot(results[0])
        

        resized_image=cv2.resize(annotated_frame, (540,284 ))
        cv2.imshow("YOLOv8 Inference", resized_image)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        
        break

cap.release()
cv2.destroyAllWindows()