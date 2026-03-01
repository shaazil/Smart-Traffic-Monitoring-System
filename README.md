# 🚗 Vehicle Analysis Dashboard

A computer vision web application built with Flask that performs **vehicle detection**, **speed estimation**, **road type classification**, and **number plate recognition** from images and videos — all running locally with no data stored.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![YOLO](https://img.shields.io/badge/YOLO-v11-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📸 Preview

> [Render Deployment](https://smart-traffic-monitoring.onrender.com)

---

## ✨ Features

### Module 01 — Vehicle Detection & Analysis
- Detects vehicles in **images and videos** using YOLOv11
- Classifies vehicle type (car, truck, bus, motorcycle, van, etc.) using **ResNet50** fine-tuned on ImageNet vehicle classes
- Displays bounding boxes, vehicle type, confidence score, and pixel dimensions
- Download annotated results directly from the UI

### Module 02 — Speed Estimation & Road Type
- Tracks vehicles frame-by-frame and estimates speed in **km/h**
- Flags **speed violations** above 70 km/h in red
- Classifies road type based on vehicle density:
  - ≤ 20 vehicles → **Highway**
  - 21–50 vehicles → **Urban Road**
  - > 50 vehicles → **Rural Road**
- Streams processed video live in the browser

### Module 03 — Number Plate Detection
- Detects number plates using a **YOLO model fine-tuned for license plates**
- Reads plate text using **EasyOCR** with preprocessing (bilateral filter, Otsu threshold)
- Validates and parses **Indian plate format**: `XX 00 XX 0000`
  - Extracts state code, district code, series, and number
  - Maps state code to full state name (MH → Maharashtra, etc.)
- For videos: captures a **vehicle snapshot** per unique plate and lists all detected plates with frame number

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Detection | [Ultralytics YOLO v11](https://github.com/ultralytics/ultralytics) |
| Classification | PyTorch, ResNet50 (torchvision) |
| Plate Detection | YOLOv8 fine-tuned (license plates) |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Image Processing | OpenCV, Pillow |
| Frontend | HTML, CSS, Vanilla JS (matrix rain UI) |

---

## Branches

- [main (Cloud-optimized deployment (Render / free-tier friendly))](https://github.com/shaazil/Smart-Traffic-Monitoring-System/tree/main)
- [local-dev (full-performance)](https://github.com/shaazil/Smart-Traffic-Monitoring-System/tree/local-dev)

---

## 📁 Project Structure

```
vehicle-analysis-dashboard/
│
├── app.py                  # Main Flask app — all routes and logic
│
├── templates/
│   ├── index.html          # Dashboard home
│   ├── detect.html         # Vehicle detection page
│   ├── speed_road.html     # Speed & road analysis page
│   └── plates.html         # Number plate detection page
│
├── static/
│   ├── styles.css          # Shared styles (used by detect.html)
│   ├── uploads/            # Temp uploaded files
│   └── results/            # Processed output files
│
├── yolo11n.pt                      # YOLOv11 nano model
├── license_plate_detector.pt       # Fine-tuned plate detection model
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/shaazil/vehicle-analysis-dashboard.git
cd vehicle-analysis-dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download models

**YOLOv11** (auto-downloads on first run via ultralytics):
```bash
# yolo11n.pt downloads automatically when you run the app
```

**License Plate Detection model** — download manually from Hugging Face:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='keremberke/yolov8n-license-plate-detection',
    filename='best.pt',
    local_dir='.'
)
import os; os.rename('best.pt', 'license_plate_detector.pt')
"
```

### 5. Run the app
```bash
python app.py
```

Visit `http://localhost:5004` in your browser.

---

## 📦 Requirements

```
flask
opencv-python
numpy
ultralytics
torch
torchvision
Pillow
easyocr
huggingface_hub
```

> Full pinned versions in `requirements.txt`. Recommended Python: **3.9–3.11**

---

## 🔧 Configuration

You can tweak these constants at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `speed_limit` | `70` | Speed violation threshold (km/h) |
| `scaling_factor` | `0.2` | Pixel-to-km/h conversion factor |
| `frame_buffer_size` | `10` | Frames used for speed smoothing |
| `highway_density_threshold` | `20` | Max vehicles for highway classification |
| `urban_density_threshold` | `50` | Max vehicles for urban classification |

---

## ⚠️ Known Limitations

- **Speed estimation** is approximate — based on pixel displacement, not real-world calibration. Accurate relative speeds but not absolute.
- **OCR accuracy** depends on plate resolution and angle. Works best on clear, front-facing plates.
- **ResNet50 classification** uses ImageNet class IDs mapped to vehicle types — may return `unknown` for some crops.
- Video processing for the detection module saves files temporarily to disk; number plate detection is fully in-memory.

---

## 🤝 Contributing

Pull requests are welcome! If you find a bug or want to add a feature (e.g. better tracking with ByteTrack, fine-tuned classifier, multi-language plates), feel free to open an issue.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO
- [keremberke](https://huggingface.co/keremberke) for the license plate detection model on Hugging Face
- [JaidedAI](https://github.com/JaidedAI/EasyOCR) for EasyOCR
- [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/) for ResNet50
