from super_gradients.training import models
from super_gradients.training.utils.media.image import load_image
import cv2
import numpy as np

# Load YOLO-NAS Nano pretrained on COCO
model = models.get("yolo_nas_n", pretrained_weights="coco")

# Class indices
PERSON_CLASS = 0
CELLPHONE_CLASS = 67  # COCO class ID for cellphone

def detect_persons_and_phones(frame, conf_thresh=0.3):
    # Convert BGR (OpenCV) to RGB (PIL-style)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    predictions = model.predict(rgb_frame, conf=conf_thresh)[0]

    persons, phones = [], []

    for pred in predictions.prediction:
        cls = int(pred.class_id)
        conf = float(pred.score)
        x1, y1, x2, y2 = map(int, pred.bbox)

        bbox = [x1, y1, x2 - x1, y2 - y1]

        if cls == PERSON_CLASS and conf > conf_thresh:
            persons.append((bbox, conf))
        elif cls == CELLPHONE_CLASS and conf > conf_thresh:
            phones.append((bbox, conf))

    return persons, phones
