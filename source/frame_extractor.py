import cv2
import threading
import queue

def extract_frames_to_queue(video_path, output_queue, interval=1):
    def _worker():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            output_queue.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval) if interval > 0 else 1
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if interval == 0 or frame_count % frame_interval == 0:
                output_queue.put(frame.copy())

            frame_count += 1

        cap.release()
        output_queue.put(None)

    thread = threading.Thread(target=_worker)
    thread.start()
    return thread
