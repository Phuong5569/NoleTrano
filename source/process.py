import cv2
import queue
import torch
import os
from datetime import datetime
from frame_extractor import extract_frames_to_queue
from detection import detect_persons_and_phones
from tracking import tracker, annotate_tracked
from pose_est import model_pose, detect_sleepy_students, keypoints_to_bbox, compute_iou

# Create log directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_EVENT = os.path.join(LOG_DIR, "event_log.txt")
LOG_SLEEP = os.path.join(LOG_DIR, "sleep.txt")
LOG_PHONE = os.path.join(LOG_DIR, "phone.txt")

def log_event(message, category=None, frame=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with open(LOG_EVENT, "a") as f:
        f.write(line)
    if category == "sleep":
        with open(LOG_SLEEP, "a") as f:
            f.write(line)
    elif category == "phone":
        with open(LOG_PHONE, "a") as f:
            f.write(line)

    # Save screenshot
    if frame is not None:
        day_folder = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(day_folder, exist_ok=True)
        file_name = f"{category}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
        path = os.path.join(day_folder, file_name)
        cv2.imwrite(path, frame)

def main():
    video_path = "data/test6.MOV"
    frame_queue = queue.Queue()
    interval = 1
    extractor_thread = extract_frames_to_queue(video_path, frame_queue, interval=interval)

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        frame = cv2.resize(frame, (960, 540))

        # --- Detection ---
        persons, phones = detect_persons_and_phones(frame)
        detections = [(box, conf, 'person') for box, conf in persons]
        tracks = tracker.update_tracks(detections, frame=frame)

        # --- Pose estimation ---
        results_pose = model_pose.predict(frame)
        poses = results_pose[0].keypoints.data if results_pose and results_pose[0].keypoints is not None else []

        matched_keypoints = []
        matched_ids = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tx1, ty1, tw, th = track.to_tlwh()
            tbox = [tx1, ty1, tx1 + tw, ty1 + th]
            best_iou = 0
            best_kp = None

            for kp in poses:
                pbox = keypoints_to_bbox(kp)
                if pbox is None:
                    continue
                iou = compute_iou(tbox, pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_kp = kp

            if best_kp is not None and best_iou > 0.3:
                matched_keypoints.append(best_kp)
                matched_ids.append(track.track_id)

        # --- Sleepy check (no timing, just pose-based) ---
        sleepy_ids = detect_sleepy_students(matched_keypoints, matched_ids)

        # --- Annotate and Log ---
        annotated = annotate_tracked(frame, tracks, phones)
        for track in tracks:
            if track.track_id in sleepy_ids:
                x, y, w, h = track.to_tlwh()
                cv2.putText(frame, "SLEEPY", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                log_event(f"Track {track.track_id} is SLEEPY", category="sleep", frame=frame)

        for phone_box, _ in phones:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x, y, w, h = track.to_tlwh()
                if any(cv2.pointPolygonTest(
                        cv2.boxPoints(((x + w/2, y + h/2), (w, h), 0)).astype(int),
                        ((phone_box[0] + phone_box[2]//2, phone_box[1] + phone_box[3]//2)), False) >= 0
                       for phone_box, _ in phones):
                    log_event(f"Track {track.track_id} is using PHONE", category="phone", frame=frame)

        cv2.imshow("Tracked + Phone + Sleepy", frame)
        key = cv2.waitKey(1 if interval == 0 else 1000)
        if key & 0xFF == ord('q'):
            break

    extractor_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
