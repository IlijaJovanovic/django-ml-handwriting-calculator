# training/export/export_model.py

from pathlib import Path
import shutil


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = PROJECT_ROOT / "math_cnn_v1.keras"
TARGET_DIR = PROJECT_ROOT / "web" / "ml_models"
TARGET_MODEL = TARGET_DIR / "math_cnn_v1.keras"


# =========================
# EXPORT LOGIC
# =========================

def main():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {SOURCE_MODEL}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_MODEL, TARGET_MODEL)

    print("[DONE] Model exported successfully")
    print(f"Source: {SOURCE_MODEL}")
    print(f"Target: {TARGET_MODEL}")


if __name__ == "__main__":
    main()
