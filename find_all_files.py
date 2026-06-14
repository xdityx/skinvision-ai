import os
from pathlib import Path

root = Path(r"C:\Users\Aditya\OneDrive\文件\Desktop\Projects\FaceRecogintion\data\raw\ACNE04")
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

non_images = []
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        p = Path(dirpath) / f
        if p.suffix.lower() not in image_extensions:
            non_images.append(p)

print(f"Total non-image files found: {len(non_images)}")
for p in sorted(non_images):
    print(p.relative_to(root.parent.parent.parent))
