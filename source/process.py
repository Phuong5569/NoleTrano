import cv2
import queue
import torch
from frame_extractor import extract_frames_to_queue
from detection import detect_persons_and_phones
from tracking import tracker, annotate_tracked
from pose_est import model_pose, check_sleepy, keypoints_to_bbox, compute_iou

import logging
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.ERROR)


def main():
    video_path = "data/test4.MOV"
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
        keypoints_list = []
        if results_pose and results_pose[0].keypoints is not None:
            keypoints_list = results_pose[0].keypoints.data  # tensor [N, 17, 3]

        matched_keypoints = []
        matched_ids = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tx1, ty1, tw, th = track.to_tlwh() 
            tbox = [tx1, ty1, tx1 + tw, ty1 + th]
            best_iou = 0
            best_kp = None

            for kp in keypoints_list:
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

        # --- Sleepy check ---
        sleepy_ids = check_sleepy(matched_keypoints, matched_ids)

        # --- Annotate and show ---
        annotated = annotate_tracked(frame, tracks, phones)
        for track in tracks:
            if track.track_id in sleepy_ids:
                x, y, w, h = track.box
                cv2.putText(frame, "SLEEPY", (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Tracked + Phone + Sleepy", frame)
        key = cv2.waitKey(1 if interval == 0 else 1000)
        if key & 0xFF == ord('q'):
            break

    extractor_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()