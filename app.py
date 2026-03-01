from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
from collections import deque
import math
from ultralytics import YOLO
import os
from PIL import Image
from torchvision import models, transforms
import torch
import tempfile
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER  = 'static/results'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER,  exist_ok=True)

# ── To save memory and easier web hosting(free) (could be removed while running loaclly) ─────────
torch.set_num_threads(1)   # HUGE memory saver

model = None
plate_model = None
classifier_model = None
# ── Models ─(define here itself while running locally for better performace) ────────
# model       = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))
# plate_model = YOLO(os.path.join(BASE_DIR, "license_plate_detector.pt"))

# classifier_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# classifier_model.eval()

# ── Vehicle class map (ResNet50 ImageNet IDs) ─────────────────────────────────
vehicles = {
    751:'car', 656:'truck', 659:'bus', 841:'car', 570:'train', 569:'truck',
    565:'truck', 737:'bicycle', 663:'motorcycle', 645:'airplane', 1065:'van',
    468:'car', 407:'car', 807:'car', 817:'car', 482:'car', 654:'car',
    832:'car', 665:'motorcycle', 802:'car', 535:'truck', 671:'car', 608:'car',
    518:'car', 670:'car', 456:'car', 691:'motorcycle', 465:'motorcycle', 539:'car',
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Speed / road constants ─────────────────────────────────────────────────────
speed_limit               = 70
scaling_factor            = 0.2
frame_buffer_size         = 10
highway_density_threshold = 20
urban_density_threshold   = 50
vehicle_positions         = {}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers #for easing memory
# ─────────────────────────────────────────────────────────────────────────────
def load_models(need_vehicle=False, need_plate=False, need_classifier=False):
    global model, plate_model, classifier_model

    if need_vehicle and model is None:
        print("Loading YOLO vehicle model...")
        model = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))

    if need_plate and plate_model is None:
        print("Loading YOLO plate model...")
        plate_model = YOLO(os.path.join(BASE_DIR, "license_plate_detector.pt"))

    if need_classifier and classifier_model is None:
        print("Loading ResNet50 classifier...")
        classifier_model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        classifier_model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — vehicle detection
# ─────────────────────────────────────────────────────────────────────────────

def classify_vehicle(image, box):
    load_models(need_classifier=True)

    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        out = classifier_model(transform(pil).unsqueeze(0))
    return vehicles.get(out.max(1)[1].item(), 'unknown')


def estimate_dimensions(box, img_h, img_w):
    x1, y1, x2, y2 = box
    if x2 > x1 and y2 > y1:
        return {'length': (x2-x1)*img_w, 'height': (y2-y1)*img_h}
    return {'length': 'undefined', 'height': 'undefined'}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — speed / road
# ─────────────────────────────────────────────────────────────────────────────

def estimate_average_speed(buf, fps, sf):
    if len(buf) < 2:
        return 0
    dist = sum(math.sqrt((buf[i][0]-buf[i-1][0])**2 + (buf[i][1]-buf[i-1][1])**2)
               for i in range(1, len(buf)))
    return (dist / (len(buf)-1)) * fps * sf


def determine_road_type(count):
    if count <= highway_density_threshold: return "Highway"
    if count <= urban_density_threshold:   return "Urban Road"
    return "Rural Road"


def generate_frames(input_path):
    load_models(need_vehicle=True) # for easing memory
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vehicle_count = 0
        for box in model(frame)[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) == 2 and float(box.conf[0]) > 0.5:
                vehicle_count += 1
                cx, cy = (x1+x2)//2, (y1+y2)//2
                vid = hash((cx//20, cy//20))
                if vid not in vehicle_positions:
                    vehicle_positions[vid] = {
                        'position_buffer': deque(maxlen=frame_buffer_size),
                        'average_speeds':  deque(maxlen=frame_buffer_size)
                    }
                vehicle_positions[vid]['position_buffer'].append((cx, cy))
                spd = estimate_average_speed(vehicle_positions[vid]['position_buffer'], fps, scaling_factor)
                vehicle_positions[vid]['average_speeds'].append(spd)
                smoothed = np.mean(vehicle_positions[vid]['average_speeds'])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                col = (0,0,255) if smoothed > speed_limit else (255,0,0)
                cv2.putText(frame, f"{smoothed:.2f} km/h", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                if smoothed > speed_limit:
                    cv2.putText(frame, "SPEED VIOLATION!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(frame, f"Road Type: {determine_road_type(vehicle_count)}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        ret, buf = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n\r\n')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — plate cropping  (NO easyocr — OCR runs in browser via Tesseract.js)
# ─────────────────────────────────────────────────────────────────────────────

def crop_plate(image, box, pad=6):
    """Return a preprocessed plate crop as base64 JPEG string."""
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w, x2+pad); y2 = min(h, y2+pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Preprocess to help Tesseract
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.bilateralFilter(gray, 11, 17, 17)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, buf = cv2.imencode('.jpg', th)
    return base64.b64encode(buf).decode('utf-8')


def vehicle_snapshot(frame, x1, y1, x2, y2):
    """Wider crop around plate to show the vehicle."""
    h, w = frame.shape[:2]
    sx1 = max(0, x1-50); sy1 = max(0, y1-130)
    sx2 = min(w, x2+50); sy2 = min(h, y2+25)
    snap = frame[sy1:sy2, sx1:sx2]
    if snap.size == 0:
        return None
    _, buf = cv2.imencode('.jpg', snap)
    return base64.b64encode(buf).decode('utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect')
def detect():
    return render_template('det.html')


@app.route('/detect/process', methods=['POST'])
def process_file():
    load_models(need_vehicle=True, need_classifier=True)#for easing memory
    file = request.files.get('file')
    if not file:
        return 'No file uploaded', 400

    input_path  = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(RESULT_FOLDER, f'processed_{file.filename}')
    file.save(input_path)

    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        image   = cv2.imread(input_path)
        results = model(image)
        detections = []
        for result in results:
            for box in result.boxes:
                if float(box.conf) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vtype = classify_vehicle(image, (x1,y1,x2,y2))
                    dims  = estimate_dimensions((x1,y1,x2,y2), image.shape[0], image.shape[1])
                    detections.append({'type': vtype, 'confidence': float(box.conf), 'dimensions': dims, 'box': [x1,y1,x2,y2]})
                    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(image, f"{vtype} ({dims['length']}px x {dims['height']}px)",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imwrite(output_path, image)
        return jsonify({'result_file': f'processed_{file.filename}', 'type': 'image', 'detections': detections})

    elif file.filename.lower().endswith(('.mp4', '.avi')):
        cap    = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        all_detections = []
        frame_count    = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            for result in model(frame):
                for box in result.boxes:
                    if float(box.conf) > 0.5:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        vtype = classify_vehicle(frame, (x1,y1,x2,y2))
                        dims  = estimate_dimensions((x1,y1,x2,y2), frame.shape[0], frame.shape[1])
                        if vtype != 'unknown':
                            all_detections.append({'type':vtype,'confidence':float(box.conf),'dimensions':dims,'frame':frame_count,'box':[x1,y1,x2,y2]})
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,f"{vtype} ({dims['length']}px x {dims['height']}px)",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            out.write(frame)
        cap.release(); out.release()
        return jsonify({'result_file': f'processed_{file.filename}', 'type': 'video', 'detections': all_detections})

    return 'Unsupported file type', 400


@app.route('/detect/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)


@app.route('/speed_road', methods=['GET', 'POST'])
def speed_road():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return 'No file uploaded', 400
        if file.filename.lower().endswith(('.mp4', '.avi')):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return render_template('speed_road.html', filename=file.filename)
    return render_template('speed_road.html')


@app.route('/speed_road/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_frames(os.path.join(UPLOAD_FOLDER, filename)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Plate routes ──────────────────────────────────────────────────────────────

@app.route('/plates')
def plates():
    return render_template('plates.html')


@app.route('/plates/process', methods=['POST'])
def process_plates():
    """
    Backend: YOLO detects plate bounding boxes only.
    Returns cropped + preprocessed plate images as base64 so Tesseract.js
    can run OCR entirely in the browser. No easyocr dependency needed.
    """
    load_models(need_plate=True) #for easing memory
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    raw_bytes  = np.frombuffer(file.read(), np.uint8)
    fname      = file.filename.lower()

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    if fname.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        image   = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        results = plate_model(image)
        detections = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < 0.35:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_b64    = crop_plate(image, (x1,y1,x2,y2))
                snapshot_b64 = vehicle_snapshot(image, x1, y1, x2, y2)

                # Draw box on annotated image (no text — JS fills it in)
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,255), 2)

                detections.append({
                    'confidence':  round(conf, 2),
                    'box':         [x1, y1, x2, y2],
                    'plate_crop':  plate_b64,    # preprocessed crop → Tesseract.js
                    'snapshot':    snapshot_b64  # vehicle photo for the card
                })

        _, buf    = cv2.imencode('.jpg', image)
        img_b64   = base64.b64encode(buf).decode('utf-8')

        return jsonify({'type': 'image', 'image': img_b64, 'detections': detections})

    # ── VIDEO ─────────────────────────────────────────────────────────────────
    elif fname.endswith(('.mp4', '.avi')):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        cap         = cv2.VideoCapture(tmp_path)
        all_detections = []
        seen_boxes  = []   # track unique plates by box position
        frame_count = 0
        SKIP        = 3    # process every Nth frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % SKIP != 0:
                continue

            for result in plate_model(frame):
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.35:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Deduplicate by checking if a very similar box was already seen
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    duplicate = any(abs(cx-px) < 40 and abs(cy-py) < 40 for px,py in seen_boxes)
                    if duplicate:
                        continue
                    seen_boxes.append((cx, cy))

                    plate_b64    = crop_plate(frame, (x1,y1,x2,y2))
                    snapshot_b64 = vehicle_snapshot(frame, x1, y1, x2, y2)

                    all_detections.append({
                        'confidence': round(conf, 2),
                        'frame':      frame_count,
                        'box':        [x1, y1, x2, y2],
                        'plate_crop': plate_b64,
                        'snapshot':   snapshot_b64
                    })

        cap.release()
        os.unlink(tmp_path)

        return jsonify({'type': 'video', 'detections': all_detections})

    return jsonify({'error': 'Unsupported file type'}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)