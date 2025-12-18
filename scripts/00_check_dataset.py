from pathlib import Path
from collections import defaultdict
from face_recognition.config import get_paths

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    paths = get_paths()
    raw_known = paths.data_dir / "raw" / "known"

    assert raw_known.exists(), f"Missing folder: {raw_known}"

    per_person = defaultdict(int)
    total_images = 0

    for person_dir in sorted(p for p in raw_known.iterdir() if p.is_dir()):
        images = [
            f for f in person_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        ]
        per_person[person_dir.name] = len(images)
        total_images += len(images)

    print("=== Dataset Summary ===")
    for person, count in per_person.items():
        print(f"{person:15s} -> {count} images")

    print("\nTotal people:", len(per_person))
    print("Total images:", total_images)

    low_count = [p for p, c in per_person.items() if c < 10]
    if low_count:
        print("\n⚠️ Warning: low image count for:", low_count)

if __name__ == "__main__":
    main()
