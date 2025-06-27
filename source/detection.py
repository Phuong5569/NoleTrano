from ultralytics import NAS



# Load YOLO-NAS model
model = NAS("model/yolo_nas_s.pt").to("cuda")  # Must support COCO class 0 = person, 67 = cellphone

def detect_persons_and_phones(frame, conf_thresh=0.3):
    results = model(frame, imgsz=640,verbose=False)[0]
    persons, phones = [], []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2 - x1, y2 - y1]

        if cls == 0 and conf > conf_thresh:
            persons.append((bbox, conf))
        elif cls == 67 and conf > conf_thresh:
            phones.append((bbox, conf))

    return persons, phones
