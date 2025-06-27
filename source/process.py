import cv2
import queue
from frame_extractor import extract_frames_to_queue
from detection import detect_persons_and_phones
from tracking import tracker, annotate_tracked

def main():
    video_path = "data/test4.MOV"
    frame_queue = queue.Queue()
    interval=0
    extractor_thread = extract_frames_to_queue(video_path, frame_queue, interval=interval)

    while True:
        frame = frame_queue.get()
        frame = cv2.resize(frame, (960, 540))
        if frame is None:
            break

        persons, phones = detect_persons_and_phones(frame)

        # Convert to DeepSORT format: ([x, y, w, h], conf, label)
        detections = [(box, conf, 'person') for box, conf in persons]

        # Track
        tracks = tracker.update_tracks(detections, frame=frame)

        # Annotate with phone proximity
        annotated = annotate_tracked(frame, tracks, phones)
          # width, height
        cv2.imshow("Tracked + Phone Proximity", frame)
        key = cv2.waitKey(1 if interval == 0 else 1000)
        if key & 0xFF == ord('q'):
            break

    extractor_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
