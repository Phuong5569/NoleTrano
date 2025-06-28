from ultralytics import YOLO
import torch

# Load YOLO-Pose model
model_pose = YOLO("model/yolov8n-pose.pt").to("cuda")

def estimate_pose(frame):
    """Run pose estimation and return list of keypoints per person"""
    results = model_pose(frame, imgsz=640, verbose=False)[0]
    return results.keypoints  # shape: (N, 17, 3)

def is_head_down(keypoints):
    """
    Determine if a person's head is down.
    keypoints: torch.Tensor (17, 3)
    Returns: True if nose below shoulders
    """
    nose = keypoints[0]
    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]

    # Check confidence
    if nose[2] < 0.5 or l_shoulder[2] < 0.5 or r_shoulder[2] < 0.5:
        return False  # not reliable

    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    return nose[1] > shoulder_y + 20  # Head is below shoulders

def detect_sleepy_students(keypoints_list, track_ids):
    """
    Given list of keypoints and IDs, return list of sleepy students (IDs).
    """
    sleepy_ids = []
    for keypoints, tid in zip(keypoints_list, track_ids):
        if is_head_down(keypoints):
            sleepy_ids.append(tid)
    return sleepy_ids
def keypoints_to_bbox(keypoints: torch.Tensor, conf_thresh=0.3):
    """
    Convert keypoints (17, 3) to bounding box (x1, y1, x2, y2).
    keypoints: torch.Tensor
    Returns: [x1, y1, x2, y2] or None if all keypoints are low confidence.
    """
    if keypoints.ndim == 2:
        valid = keypoints[:, 2] > conf_thresh
        if valid.sum() == 0:
            return None
        coords = keypoints[valid][:, :2]
        x1, y1 = coords.min(dim=0).values
        x2, y2 = coords.max(dim=0).values
        return [x1.item(), y1.item(), x2.item(), y2.item()]
    return None

def compute_iou(boxA, boxB):
    """
    Compute IoU between two bounding boxes.
    Each box is [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    iou = interArea / (union + 1e-6)
    return iou
