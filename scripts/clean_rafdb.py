import os
import sys
from pathlib import Path
import cv2

# Ensure project root is on sys.path when running from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def clean_folder(root_dir: str, *, min_size: int = 48) -> dict:
    """
    Removes:
    - unreadable/corrupt images
    - very small images (min side < min_size)
    Returns stats dict.
    """
    valid_ext = (".jpg", ".jpeg", ".png")
    removed = 0
    checked = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(valid_ext):
                continue
            checked += 1
            path = os.path.join(dirpath, name)
            try:
                img = cv2.imread(path)
                if img is None:
                    os.remove(path)
                    removed += 1
                    continue
                h, w = img.shape[:2]
                if min(h, w) < min_size:
                    os.remove(path)
                    removed += 1
            except Exception:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass

    return {"checked": checked, "removed": removed, "kept": checked - removed}


def main():
    base_dir = os.environ.get("RAF_DB_DIR", "/home/ns-44/Desktop/Mood Detector/raf_db_extracted")
    train_dir = os.path.join(base_dir, "train_data")
    test_dir = os.path.join(base_dir, "test_data")
    min_size = int(os.environ.get("MIN_SIZE", "48"))

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise SystemExit(f"Expected folders not found: {train_dir} and {test_dir}")

    print(f"Cleaning RAF-DB at: {base_dir}")
    print(f"Min image side: {min_size}px")

    s1 = clean_folder(train_dir, min_size=min_size)
    s2 = clean_folder(test_dir, min_size=min_size)

    print(f"train_data: checked={s1['checked']} removed={s1['removed']} kept={s1['kept']}")
    print(f"test_data : checked={s2['checked']} removed={s2['removed']} kept={s2['kept']}")


if __name__ == "__main__":
    main()

