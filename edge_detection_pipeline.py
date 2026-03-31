# ==========================================
# Edge Detection Pipeline
# Choose: Upload Image  OR  Live Camera
# Detectors: Canny, Sobel, Laplacian,
#             Prewitt, Scharr
# ==========================================

import os
import sys
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
DEFAULT_IMAGE_URL = "https://images.unsplash.com/photo-1503376780353-7e6692767b70"
OUTPUT_IMG_DIR    = "outputs/images"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs("sample_data", exist_ok=True)


# ==========================================
# EDGE DETECTION METHODS
# ==========================================
def apply_blur(gray):
    return cv2.GaussianBlur(gray, (5, 5), 1.5)


def sobel_edges(img):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))


def laplacian_edges(img):
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    return cv2.convertScaleAbs(lap)


def canny_edges(img):
    return cv2.Canny(img, 50, 150)


def prewitt_edges(img):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(img.astype(np.float32), -1, kx)
    py = cv2.filter2D(img.astype(np.float32), -1, ky)
    return np.uint8(np.clip(np.sqrt(px**2 + py**2), 0, 255))


def scharr_edges(img):
    sx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    return np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))


def run_all_detectors(blurred):
    return {
        "Canny":     canny_edges(blurred),
        "Sobel":     sobel_edges(blurred),
        "Laplacian": laplacian_edges(blurred),
        "Prewitt":   prewitt_edges(blurred),
        "Scharr":    scharr_edges(blurred),
    }


# ==========================================
# HELPER — add label to a frame tile
# ==========================================
def add_label(tile, text):
    """Draw a dark banner with white label at the top of a BGR tile."""
    labeled = tile.copy()
    cv2.rectangle(labeled, (0, 0), (tile.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(labeled, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


# ==========================================
# SAVE OUTPUTS (image mode)
# ==========================================
def save_outputs(gray, rgb, detectors):
    all_panels = {"Original": gray, "Blur": apply_blur(gray), **detectors}

    # Comparison grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Edge Detection — All Methods", fontsize=16, fontweight="bold")
    for ax, (name, img) in zip(axes.flat, all_panels.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=13)
        ax.axis("off")
    axes.flat[7].set_visible(False)
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_IMG_DIR, "comparison_all.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"  Saved: {grid_path}")

    # Canny green overlay
    overlay = rgb.copy()
    overlay[detectors["Canny"] > 0] = [0, 255, 0]
    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.title("Canny Edge Overlay (green)", fontsize=13)
    plt.axis("off")
    overlay_path = os.path.join(OUTPUT_IMG_DIR, "overlay_canny.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"  Saved: {overlay_path}")

    # Individual saves
    for name, img in detectors.items():
        path = os.path.join(OUTPUT_IMG_DIR, f"{name.lower()}.png")
        plt.figure(figsize=(8, 5))
        plt.imshow(img, cmap="gray")
        plt.title(f"{name} Edge Detection", fontsize=13)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_IMG_DIR)}\n")


# ==========================================
# MODE 1 — UPLOAD / LOCAL IMAGE
# ==========================================
def mode_image():
    print("\n" + "="*52)
    print("  IMAGE MODE")
    print("="*52)
    print("\nOptions:")
    print("  [1] Use a local file path   (e.g. /home/user/photo.jpg)")
    print("  [2] Download sample image   (Unsplash car photo)")

    sub = input("\nEnter 1 or 2: ").strip()

    if sub == "1":
        path = input("Enter full path to your image: ").strip().strip('"').strip("'")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        bgr = cv2.imread(path)
        if bgr is None:
            print("Could not read image. Check path and format.")
            return
        print(f"Loaded: {path}")
    else:
        dest = "sample_data/sample.jpg"
        if not os.path.exists(dest):
            print("Downloading sample image...")
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(DEFAULT_IMAGE_URL, headers=headers, stream=True)
            if r.status_code != 200:
                print(f"Download failed (HTTP {r.status_code}).")
                return
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("Download complete.")
        else:
            print("Sample image already exists.")
        bgr = cv2.imread(dest)

    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    blurred = apply_blur(gray)

    print("\nRunning all 5 edge detectors...")
    detectors = run_all_detectors(blurred)
    print("Saving outputs...\n")
    save_outputs(gray, rgb, detectors)


# ==========================================
# MODE 2 — LIVE CAMERA (ALL 5 at once)
# ==========================================
def mode_live_camera():
    print("\n" + "="*52)
    print("  LIVE CAMERA MODE  —  All 5 Detectors")
    print("="*52)
    print("\nOpening webcam...")
    print("Layout:  Original | Canny | Sobel")
    print("         Laplacian | Prewitt | Scharr")
    print("\nPress Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible. Check if it is connected and not in use.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        # Resize frame for a comfortable grid (each tile ~426x240 for 1280 wide)
        h, w = frame.shape[:2]
        tile_w = 426
        tile_h = int(h * tile_w / w)
        small  = cv2.resize(frame, (tile_w, tile_h))

        gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = apply_blur(gray)

        # Run all detectors
        results = {
            "Canny":     canny_edges(blurred),
            "Sobel":     sobel_edges(blurred),
            "Laplacian": laplacian_edges(blurred),
            "Prewitt":   prewitt_edges(blurred),
            "Scharr":    scharr_edges(blurred),
        }

        # Convert all edge maps to BGR for stacking
        tiles = {"Original": small}
        for name, edges in results.items():
            tiles[name] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Add labels to each tile
        labeled = {name: add_label(tile, name) for name, tile in tiles.items()}

        # Build 2-row grid: [Original | Canny | Sobel]
        #                   [Laplacian | Prewitt | Scharr]
        row1 = np.hstack([labeled["Original"], labeled["Canny"],  labeled["Sobel"]])
        row2 = np.hstack([labeled["Laplacian"], labeled["Prewitt"], labeled["Scharr"]])
        grid = np.vstack([row1, row2])

        # Title bar at the very top
        title_bar = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, "Edge Detection  |  All 5 Methods  |  Press Q to quit",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        display = np.vstack([title_bar, grid])

        cv2.imshow("Edge Detection — Live", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


# ==========================================
# MAIN MENU
# ==========================================
def main():
    print("\n" + "="*52)
    print("  EDGE DETECTION PIPELINE")
    print("  Canny | Sobel | Laplacian | Prewitt | Scharr")
    print("="*52)
    print("\nSelect mode:")
    print("  [1] Upload / Local image")
    print("  [2] Live camera  (shows all 5 detectors at once)")
    print("  [3] Both  (image first, then camera)")
    print("  [0] Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        mode_image()
    elif choice == "2":
        mode_live_camera()
    elif choice == "3":
        mode_image()
        again = input("Launch live camera now? (y/n): ").strip().lower()
        if again == "y":
            mode_live_camera()
    elif choice == "0":
        print("Bye!")
        sys.exit(0)
    else:
        print("Invalid choice. Run the script again and enter 1, 2, 3, or 0.")


if __name__ == "__main__":
    main()
