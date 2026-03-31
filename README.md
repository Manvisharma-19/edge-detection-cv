# 🎯 Real-Time Edge Detection using OpenCV

> **Course:** Computer Vision — Bring Your Own Project (BYOP)
> **Deadline:** March 31, 2026
> **Submitted to:** Dr. Rajneesh Patel — VITyarthi Portal
> **Aim:** Implementation of Computer Vision in Real Life

A complete edge detection pipeline built using OpenCV that processes static images and live webcam input to highlight object boundaries using classical computer vision techniques.

---

## 📌 Problem Statement

Detecting object boundaries is a fundamental step in many computer vision tasks such as object recognition, segmentation, and tracking. Manual identification is inefficient and inconsistent.

This project solves that by building a real-time edge detection system that can:

- ✅ Detect edges in static images using multiple algorithms
- ✅ Apply real-time edge detection on live webcam feed
- ✅ Compare different edge detection techniques for better understanding
- ✅ Generate clean, processed outputs for analysis and visualization

---

## 🗂️ Project Structure

```
edge-detection-byop/
│
├── edge_detection_pipeline.py
├── requirements.txt
│
├── sample_data/
│   └── sample.jpg
│
├── outputs/
│   └── images/
│       ├── original.jpg
│       ├── canny_edges.jpg
│       ├── sobel_edges.jpg
│       ├── laplacian_edges.jpg
│       ├── prewitt_edges.jpg      
│       ├── scharr_edges.jpg
│       └── comparison.png
│
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- Webcam *(optional)*
- ~20 MB disk space

### Install via `requirements.txt`

```bash
pip install -r requirements.txt
```

### Or Install Manually

```bash
pip install opencv-python numpy matplotlib requests tqdm
```

### Create Virtual Environment *(Recommended)*

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python edge_detection_pipeline.py
```

---

## 🔄 Project Workflow

```
┌──────────────────────────────┐
│        INPUT SOURCES         │
│   Static Image | Webcam Feed │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│     Image Preprocessing      │
│  Grayscale + Noise Reduction │
└───────────────┬──────────────┘
                ▼
┌────────────────────────────────────────────────────┐
│          Edge Detection Algorithms                 │
│  Canny │ Sobel │ Laplacian │ Prewitt │ Scharr      │
└───────────────┬────────────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│             OUTPUT RESULTS                 │
│ Edge Images (All Methods) + Comparison     │
└────────────────────────────────────────────┘


```

---

## 🔍 Edge Detection Techniques Used

### 1. Canny Edge Detection
- Most widely used method
- Multi-stage algorithm
- Produces clean and thin edges

### 2. Sobel Operator
- Detects gradients in horizontal & vertical directions
- Highlights edges based on intensity changes

### 3. Laplacian Operator
- Detects edges using second-order derivatives
- Captures fine details

### 4. Prewitt Operator
-Detects edges using first-order derivatives


### 5. Scharr Operator
-Improved version of Sobel operator and produces sharper edge results



---

##📤 Upload Image Edge Detection
-Allows user to upload custom image

-Applies all edge detection techniques

-Displays comparison of results

---

## 📷 Webcam Edge Detection

- Opens live camera feed
- Applies edge detection in real-time
- Press **Q** to exit

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `original.jpg` | Input image |
| `canny_edges.jpg` | Canny output |
| `sobel_edges.jpg` | Sobel output |
| `laplacian_edges.jpg` | Laplacian output |
| `prewitt_edges.jpg` | Laplacian output |
| `scharr_edges.jpg` | Laplacian output |
| `comparison.png` | Side-by-side comparison |

---

## 🛠️ Configuration

Modify parameters inside the script:

```python
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150
```

> Adjust thresholds for better edge detection results.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image processing |
| `numpy` | Array operations |
| `matplotlib` | Visualization |
| `requests` | Download sample image |
| `tqdm` | Progress bar |

---

## 🧠 Key Concepts

- Image Gradients
- Noise Reduction (Gaussian Blur)
- Thresholding
- Edge Localization

---

## 🚧 Known Limitations

- Sensitive to noise in low-quality images
- Threshold tuning required for best results
- Real-time webcam performance depends on hardware

---

## 📄 License

This project is submitted as academic coursework for the Computer Vision course.

---

## 👤 Author

**Manvi Sharma**
`23BAI10777`
Computer Vision — BYOP Submission, March 2026
