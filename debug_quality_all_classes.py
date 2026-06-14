import pandas as pd
from pathlib import Path

csv_path = Path(r"C:\Users\Aditya\OneDrive\文件\Desktop\Projects\FaceRecogintion\data\phase0_outputs\quality_audit.csv")
df = pd.read_csv(csv_path)

for class_idx in range(4):
    c_df = df[df["severity_label"] == class_idx]
    class_name = c_df["severity_name"].iloc[0] if len(c_df) > 0 else f"Class {class_idx}"
    print(f"\n==================================================")
    print(f"Class {class_idx} ({class_name.upper()}): {len(c_df)} total images")
    print(f"==================================================")
    print(f"Passed quality pass: {len(c_df[c_df['quality_pass'] == True])}")
    print(f"Failed quality pass: {len(c_df[c_df['quality_pass'] == False])}")
    print(f"  - Corrupted: {len(c_df[c_df['is_corrupted'] == True])}")
    print(f"  - Blurry (score < 100.0): {len(c_df[c_df['is_blurry'] == True])}")
    print(f"  - Low contrast: {len(c_df[c_df['exposure_flag'] == 'low_contrast'])}")
    print(f"  - Underexposed: {len(c_df[c_df['exposure_flag'] == 'underexposed'])}")
    print(f"  - Overexposed: {len(c_df[c_df['exposure_flag'] == 'overexposed'])}")
    print(f"  - No face detected: {len(c_df[c_df['face_flag'] == 'no_face'])}")
    print(f"  - Low face confidence: {len(c_df[c_df['face_flag'] == 'low_confidence'])}")
    print(f"  - Multi-face: {len(c_df[c_df['face_flag'] == 'multi_face'])}")
