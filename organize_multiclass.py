import os
import pandas as pd
import shutil

print("Organizing dataset for multi-class classification...")
# --- Configuration ---
csv_path = 'HAM10000_metadata.csv'
source_dir = 'all_images'
destination_dir = 'data_multiclass/train' 

# --- Script Logic ---
metadata = pd.read_csv(csv_path)

lesion_types = metadata['dx'].unique()
print(f"Found {len(lesion_types)} lesion types: {lesion_types}")

for lesion_type in lesion_types:
    os.makedirs(os.path.join(destination_dir, lesion_type), exist_ok=True)

copied_count = 0
for index, row in metadata.iterrows():
    image_id = row['image_id']
    lesion_type = row['dx']
    image_filename = f"{image_id}.jpg"
    
    source_path = os.path.join(source_dir, image_filename)
    destination_path = os.path.join(destination_dir, lesion_type, image_filename)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        copied_count += 1

print(f"\nDone! Organized a total of {copied_count} images into {len(lesion_types)} folders.")