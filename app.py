from flask import Flask, render_template, Response,jsonify,request,send_file
import cv2
import numpy as np
from collections import deque
import math
from ultralytics import YOLO
import os
from PIL import Image
from torchvision import models, transforms
import torch
import easyocr
import tempfile, base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))
'''Number plate detection model'''
plate_model = YOLO(os.path.join(BASE_DIR, "license_plate_detector.pt"))
plate_model = YOLO("license_plate_detector.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False)  
'''vehicle detection model'''
classifier_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  
classifier_model.eval()  
#
vehicles = {
    751: 'car',656: 'truck',659: 'bus',841: 'car',570: 'train',569:'truck',565:'truck',737: 'bicycle',663: 'motorcycle',   
    645: 'airplane',1065: 'van',468:'car',407:'car',807:'car',817:'car',482:'car',654:'car',832:'car',665:'motorcycle',
    802: 'car',535: 'truck',671: 'car',608: 'car',518: 'car',670: 'car',456: 'car',691:'motorcycle',465:'motorcycle',539:'car',
}
#

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def classify_vehicle(image, box):
    x1, y1, x2, y2 = map(int, box)
    vehicle_img = image[y1:y2, x1:x2]  
    vehicle_img_pil = Image.fromarray(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB))  
    input_tensor = transform(vehicle_img_pil)  
    input_batch = input_tensor.unsqueeze(0) 
    
    with torch.no_grad():  
        output = classifier_model(input_batch)  

    _, predicted = output.max(1)
    predicted_class = predicted.item()
    
    vehicle_type = vehicles.get(predicted_class, 'unknown')
    return vehicle_type  

def estimate_dimensions(box, img_height, img_width):
    x1, y1, x2, y2 = box
    if x2 > x1 and y2 > y1:
        length = (x2 - x1) * img_width  
        height = (y2 - y1) * img_height  
    else:
        length = "undefined"
        height = "undefined"
    return {'length': length, 'height': height}

'''vehicle detection model'''



'''speed estimation & road type detection'''

speed_limit = 70
scaling_factor = 0.2
frame_buffer_size = 10
highway_density_threshold = 20
urban_density_threshold = 50

# Vehicle tracking dictionary
vehicle_positions = {}

def estimate_average_speed(position_buffer, fps, scaling_factor):
    if len(position_buffer) < 2:
        return 0
    total_distance = sum(
        math.sqrt((position_buffer[i][0] - position_buffer[i - 1][0]) ** 2 + 
                  (position_buffer[i][1] - position_buffer[i - 1][1]) ** 2)
        for i in range(1, len(position_buffer))
    )
    avg_speed = (total_distance / (len(position_buffer) - 1)) * fps * scaling_factor
    return avg_speed

def determine_road_type(vehicle_count):
    if vehicle_count <= highway_density_threshold:
        return "Highway"
    elif vehicle_count <= urban_density_threshold:
        return "Urban Road"
    else:
        return "Rural Road"

def generate_frames(input_path):
    cap = cv2.VideoCapture(input_path)  # Video file path
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes

        vehicle_count = 0
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == 2 and confidence > 0.5:
                vehicle_count += 1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                vehicle_id = hash((center_x // 20, center_y // 20))

                if vehicle_id not in vehicle_positions:
                    vehicle_positions[vehicle_id] = {'position_buffer': deque(maxlen=frame_buffer_size), 'average_speeds': deque(maxlen=frame_buffer_size)}

                vehicle_positions[vehicle_id]['position_buffer'].append((center_x, center_y))

                current_speed = estimate_average_speed(vehicle_positions[vehicle_id]['position_buffer'], frame_rate, scaling_factor)
                vehicle_positions[vehicle_id]['average_speeds'].append(current_speed)
                smoothed_speed = np.mean(vehicle_positions[vehicle_id]['average_speeds'])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                speed_color = (0, 0, 255) if smoothed_speed > speed_limit else (255, 0, 0)
                cv2.putText(frame, f"{smoothed_speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)

                if smoothed_speed > speed_limit:
                    cv2.putText(frame, "SPEED VIOLATION!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        road_type = determine_road_type(vehicle_count)
        cv2.putText(frame, f"Road Type: {road_type}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

'''speed estimation & road type detection'''

'''Number plate detection'''
def read_plate_text(image, box):
    x1, y1, x2, y2 = map(int, box)
    pad = 5
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)
    
    plate_crop = image[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return "unreadable"
    
    # Preprocess
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    results = ocr_reader.readtext(thresh)
    if not results:
        return "unreadable"
    
    # Sort detections left to right by x position
    results = sorted(results, key=lambda x: x[0][0][0])
    
    # Filter low confidence results and join all text
    plate_parts = [r[1].upper().strip() for r in results if r[2] > 0.25]
    
    if not plate_parts:
        return "unreadable"
    
    full_plate = ' '.join(plate_parts)
    
    # Clean up — remove spaces within plate number for standard formats
    # Indian plates follow: XX 00 X 0000 format
    # Try to find the main plate number and exclude "IND" label
    filtered = [p for p in plate_parts if p != 'IND' and len(p) > 1]
    
    if filtered:
        return ''.join(filtered)  # join without spaces e.g. WB06F5977
    return full_plate
'''Number plate detection'''


# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')  # Main page, choose options here

# Route for Vehicle Detection
@app.route('/detect')
def detect():
    return render_template('det.html') ##hereeee
 
@app.route('/detect/process', methods=['POST'])
def process_file():
    file = request.files['file']
    if not file:
        return 'No file uploaded', 400
    
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)
    
    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', 'webp')):
        image = cv2.imread(input_path)
        results = model(image)
        
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf > 0.5:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_type = classify_vehicle(image, (x1, y1, x2, y2))
                    dimensions = estimate_dimensions((x1, y1, x2, y2), image.shape[0], image.shape[1])
                    detections.append({
                        'type': vehicle_type,
                        'confidence': float(box.conf),
                        'dimensions': dimensions,
                        'box': [x1, y1, x2, y2]
                    })
                    print(detections)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{vehicle_type} ({dimensions['length']}px x {dimensions['height']}px)"
                    cv2.putText(image, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                    output_path = os.path.join(RESULT_FOLDER, f'processed_{file.filename}')
                    cv2.imwrite(output_path, image)
        
        return jsonify({
            'result_file': f'processed_{file.filename}',
            'type': 'image',
            'detections': detections
        })
        
    elif file.filename.lower().endswith(('.mp4', '.avi')):
        output_path = os.path.join(RESULT_FOLDER, f'processed_{file.filename}')
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        frame_count = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
                
            results = model(frame)
            frame_detections = []
            
            for result in results:
                for box in result.boxes:
                    if box.conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicle_type = classify_vehicle(frame, (x1, y1, x2, y2))
                        dimensions = estimate_dimensions(
                            (x1, y1, x2, y2),
                            frame.shape[0],
                            frame.shape[1]
                        )
                        if vehicle_type != 'unknown':
                            frame_detections.append({
                            'type': vehicle_type,
                            'confidence': float(box.conf),
                            'dimensions': dimensions,
                            'frame': frame_count,
                            'box': [x1, y1, x2, y2]
                        })
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{vehicle_type} ({dimensions['length']}px x {dimensions['height']}px)"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            all_detections.extend(frame_detections)
            out.write(frame)
            
        cap.release()
        out.release()
        
        return jsonify({
            'result_file': f'processed_{file.filename}',
            'type': 'video',
            'detections': all_detections
        })

@app.route('/detect/download/<filename>')
def download(filename):
    return send_file(
        os.path.join(RESULT_FOLDER, filename),
        as_attachment=True
    )


# Route for video feed
@app.route('/speed_road', methods=['GET', 'POST'])
def speed_road():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        
        if file and file.filename.lower().endswith(('.mp4', '.avi')):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return render_template('speed_road.html', filename=filename)
    
    return render_template('speed_road.html')

@app.route('/speed_road/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_frames(video_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/other_page')
def other_page():
    return render_template('other_page.html')  
##here
@app.route('/plates')
def plates():
    return render_template('plates.html')


@app.route('/plates/process', methods=['POST'])
def process_plates():
    file = request.files['file']
    if not file:
        return 'No file uploaded', 400

    file_bytes = np.frombuffer(file.read(), np.uint8)

    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = plate_model(image)

        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_text = read_plate_text(image, (x1, y1, x2, y2))
                    confidence = float(box.conf)

                    detections.append({
                        'plate_text': plate_text,
                        'confidence': round(confidence, 2),
                        'box': [x1, y1, x2, y2]
                    })

                    # Draw on image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(image, plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Return processed image directly without saving
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = __import__('base64').b64encode(buffer).decode('utf-8')

        return jsonify({
            'type': 'image',
            'image': img_base64,
            'detections': detections
        })

    elif file.filename.lower().endswith(('.mp4', '.avi')):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path.replace('.mp4', '_out.mp4')

            cap = cv2.VideoCapture(tmp_in_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_out_path, fourcc,
                                cap.get(cv2.CAP_PROP_FPS),
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            all_detections = []
            frame_count = 0
            seen_plates = {}  # plate_text -> snapshot base64

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                results = plate_model(frame)
                for result in results:
                    for box in result.boxes:
                        if box.conf > 0.4:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            plate_text = read_plate_text(frame, (x1, y1, x2, y2))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, plate_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                            # Snap the vehicle crop (wider area around plate)
                            if plate_text not in seen_plates and plate_text != "unreadable":
                                vh = frame.shape[0]
                                vw = frame.shape[1]
                                # Expand crop upward to capture vehicle
                                snap_x1 = max(0, x1 - 40)
                                snap_y1 = max(0, y1 - 120)
                                snap_x2 = min(vw, x2 + 40)
                                snap_y2 = min(vh, y2 + 20)
                                vehicle_snap = frame[snap_y1:snap_y2, snap_x1:snap_x2]
                                _, snap_buf = cv2.imencode('.jpg', vehicle_snap)
                                snap_b64 = base64.b64encode(snap_buf).decode('utf-8')

                                seen_plates[plate_text] = snap_b64
                                all_detections.append({
                                    'plate_text': plate_text,
                                    'confidence': round(float(box.conf), 2),
                                    'frame': frame_count,
                                    'snapshot': snap_b64  # vehicle photo
                                })

                out.write(frame)

            cap.release()
            out.release()
            os.unlink(tmp_in_path)

            return jsonify({
                'type': 'video',
                'detections': all_detections
            })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)