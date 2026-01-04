# training/model/train.py

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from architecture import build_model


DATASET_PATH = Path(__file__).resolve().parents[2] /"training"/ "custom_math_dataset_v1.npz"
MODEL_OUT = Path(__file__).resolve().parents[2] / "web" / "ml_models" / "math_cnn_v1.keras"


def main():
    print("[INFO] Loading dataset...")
    data = np.load(DATASET_PATH)
    X = data["X"]
    y = data["Y"]

    # reshape for CNN
    X = X[..., np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(f"[INFO] Train samples: {len(y_train)}")
    print(f"[INFO] Val samples: {len(y_val)}")

    model = build_model()

    print("[INFO] Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64
    )

    print(f"[INFO] Saving model to {MODEL_OUT}")
    model.save(MODEL_OUT)

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
