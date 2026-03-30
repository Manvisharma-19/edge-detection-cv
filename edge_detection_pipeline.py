# ==========================================
# Edge Detection Pipeline — BYOP (FINAL)
# IMAGE + LIVE CAMERA ONLY
# ==========================================

import os
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
IMAGE_URL = "https://images.unsplash.com/photo-1503376780353-7e6692767b70"
IMAGE_PATH = "sample_data/sample.jpg"

OUTPUT_IMG_DIR = "outputs/images"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs("sample_data", exist_ok=True)

# ==========================================
# DOWNLOAD IMAGE
# ==========================================
def download_image():
    if os.path.exists(IMAGE_PATH):
        print("Image already exists.")
        return

    print("Downloading image...")
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(IMAGE_URL, headers=headers, stream=True)

    if r.status_code != 200:
        raise Exception("Image download failed")

    total = int(r.headers.get("content-length", 0))

    with open(IMAGE_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(" Image downloaded!\n")


# ==========================================
# EDGE METHODS
# ==========================================
def apply_blur(gray):
    return cv2.GaussianBlur(gray, (5, 5), 1.5)


def sobel_edges(img):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(0.5*sx + 0.5*sy)


def laplacian_edges(img):
    return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))


def canny_edges(img):
    return cv2.Canny(img, 50, 150)


def prewitt_edges(img):
    kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)

    px = cv2.filter2D(img.astype(np.float32), -1, kernelx)
    py = cv2.filter2D(img.astype(np.float32), -1, kernely)

    return np.uint8(np.clip(np.sqrt(px**2 + py**2), 0, 255))


def scharr_edges(img):
    sx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    return np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))


# ==========================================
# PROCESS IMAGE
# ==========================================
def process_image():
    download_image()

    img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blur = apply_blur(gray)

    images = {
        "Original": gray,
        "Blur": blur,
        "Sobel": sobel_edges(blur),
        "Laplacian": laplacian_edges(blur),
        "Canny": canny_edges(blur),
        "Prewitt": prewitt_edges(blur),
        "Scharr": scharr_edges(blur),
    }

    # Save comparison
    plt.figure(figsize=(15, 8))
    for i, (name, img) in enumerate(images.items()):
        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, "comparison.png"))
    plt.close()

    # Overlay
    overlay = rgb.copy()
    overlay[images["Canny"] > 30] = [0, 255, 0]

    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, "overlay.png"))
    plt.close()

    print(" Image processing done!\n")


# ==========================================
# LIVE CAMERA
# ==========================================
def live_camera():
    print("Starting webcam... Press Q to exit")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = canny_edges(apply_blur(gray))

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, edges_bgr])

        cv2.imshow("Live Edge Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    process_image()
    live_camera()

    print("\n All outputs saved in /outputs")