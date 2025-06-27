from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

tracker = DeepSort(max_age=30)

def box_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def is_near(box1, box2, threshold=100):
    c1 = box_center(box1)
    c2 = box_center(box2)
    distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    return distance < threshold

def is_inside(person_box, phone_box):
    # person_box and phone_box are [x, y, w, h]
    px, py, pw, ph = person_box
    ph_x, ph_y, ph_w, ph_h = phone_box

    return (
        ph_x >= px and
        ph_y >= py and
        ph_x + ph_w <= px + pw and
        ph_y + ph_h <= py + ph
    )

def annotate_tracked(frame, tracked_persons, phones):
    for track in tracked_persons:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        w, h = r - l, b - t
        person_box = [l, t, w, h]

        near_phone = any(
            is_inside(person_box, phone_box) or is_near(person_box, phone_box, threshold=80)
            for phone_box, _ in phones
)

        color = (0, 0, 255) if near_phone else (0, 255, 0)
        label = f"ID {track_id}" + (" 📱" if near_phone else "")

        cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)
        cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Optional: show phone boxes
    for phone_box, conf in phones:
        x, y, w, h = phone_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        cv2.putText(frame, f"phone {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return frame
