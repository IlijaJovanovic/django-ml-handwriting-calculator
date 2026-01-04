import numpy as np
import struct
import os
from pathlib import Path
from PIL import Image
from sklearn.utils import shuffle


# =========================
# CONFIG
# =========================

IMAGE_SIZE = (28, 28)

LABEL_MAP = {
    "plus": 10,
    "minus": 11,
    "multiply": 12
}


# =========================
# MNIST LOADERS
# =========================

def load_mnist_images(path: Path):
    with open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def load_mnist_labels(path: Path):
    with open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# =========================
# SYMBOL LOADER
# =========================

def load_symbol_folder(folder_path: Path, label: int):
    images = []
    labels = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        img = Image.open(folder_path / fname).convert("L")
        img = img.resize(IMAGE_SIZE)

        images.append(np.array(img))
        labels.append(label)

    return images, labels


# =========================
# MAIN PIPELINE
# =========================

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    MNIST_DIR = PROJECT_ROOT / "training" / "datasets" / "mnist"
    SYMBOLS_DIR = PROJECT_ROOT / "training" / "datasets" / "symbols"
    OUTPUT_FILE = PROJECT_ROOT / "training" / "custom_math_dataset_v1.npz"

    # --- Load MNIST ---
    print("[INFO] Loading MNIST dataset...")

    x_train = load_mnist_images(MNIST_DIR / "train-images.idx3-ubyte")
    y_train = load_mnist_labels(MNIST_DIR / "train-labels.idx1-ubyte")

    x_test = load_mnist_images(MNIST_DIR / "t10k-images.idx3-ubyte")
    y_test = load_mnist_labels(MNIST_DIR / "t10k-labels.idx1-ubyte")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    print(f"[INFO] MNIST samples: {len(Y)}")

    # --- Load custom symbols ---
    custom_images = []
    custom_labels = []

    print("[INFO] Loading custom symbols...")
    for symbol, label in LABEL_MAP.items():
        folder = SYMBOLS_DIR / symbol
        imgs, lbls = load_symbol_folder(folder, label)

        custom_images.extend(imgs)
        custom_labels.extend(lbls)

        print(f"  - {symbol}: {len(imgs)} samples")

    custom_images = np.array(custom_images, dtype="float32") / 255.0
    custom_labels = np.array(custom_labels, dtype="int64")

    # --- Combine datasets ---
    X = np.concatenate((X, custom_images), axis=0)
    Y = np.concatenate((Y, custom_labels), axis=0)

    print(f"[INFO] Total samples before shuffle: {len(Y)}")

    X, Y = shuffle(X, Y, random_state=42)

    # --- Save ---
    np.savez(OUTPUT_FILE, X=X, Y=Y)

    print("[DONE] Dataset created successfully")
    print(f"[INFO] Saved to: {OUTPUT_FILE}")
    print(f"[INFO] Final shapes -> X: {X.shape}, Y: {Y.shape}")


if __name__ == "__main__":
    main()
