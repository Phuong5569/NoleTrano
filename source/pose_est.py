from ultralytics import YOLO
import time
from collections import defaultdict


model_pose = YOLO("model/yolov8n-pose.pt").to("cuda")

def estimate_pose(frame):
    results = model_pose(frame, imgsz=640,verbose=False)[0]
    return results.keypoints  # torch.Tensor [N, 17, 3]

head_down_times = defaultdict(lambda: {"start": None, "is_down": False})

def is_head_down(keypoints):
    # keypoints: tensor [17, 3] => (x, y, confidence)
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
        return False  # unreliable

    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    return nose[1] > avg_shoulder_y + 20  # threshold: nose below shoulder by 20px

def check_sleepy(keypoints_list, track_ids, sleepy_threshold=10.0):
    sleepy_ids = []

    current_time = time.time()
    for kp, tid in zip(keypoints_list, track_ids):
        down = is_head_down(kp)

        rec = head_down_times[tid]
        if down:
            if not rec["is_down"]:
                rec["start"] = current_time
                rec["is_down"] = True
            elif current_time - rec["start"] > sleepy_threshold:
                sleepy_ids.append(tid)
        else:
            rec["is_down"] = False
            rec["start"] = None

    return sleepy_ids
def keypoints_to_bbox(keypoints, conf_thresh=0.3):
    """Convert keypoints to bounding box (x1, y1, x2, y2)"""
    valid = keypoints[:, 2] > conf_thresh
    if valid.sum() == 0:
        return None
    points = keypoints[valid][:, :2]
    x1, y1 = points.min(0).values
    x2, y2 = points.max(0).values
    return [x1.item(), y1.item(), x2.item(), y2.item()]

def compute_iou(boxA, boxB):
    """IoU between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou