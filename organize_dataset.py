import os
import pandas as pd
import shutil

def organize_dataset():
    """
    Reads the HAM10000 metadata and organizes images into 'benign' and
    'malignant' folders based on their lesion type.
    """
    csv_path = 'HAM10000_metadata.csv'
    source_images_dir = 'all_images' 
    destination_dir = 'data/train'

    print("Reading metadata from:", csv_path)
    metadata = pd.read_csv(csv_path)

    benign_types = ['nv', 'bkl', 'df']
    malignant_types = ['mel', 'bcc', 'akiec', 'vasc']

    benign_dest = os.path.join(destination_dir, 'benign')
    malignant_dest = os.path.join(destination_dir, 'malignant')
    os.makedirs(benign_dest, exist_ok=True)
    os.makedirs(malignant_dest, exist_ok=True)

    print(f"Destination folders created at '{benign_dest}' and '{malignant_dest}'")

    copied_count = 0
    for index, row in metadata.iterrows():
        lesion_type = row['dx']
        image_id = row['image_id']
        image_filename = f"{image_id}.jpg"
        source_path = os.path.join(source_images_dir, image_filename)

        destination_path = None
        if lesion_type in benign_types:
            destination_path = os.path.join(benign_dest, image_filename)
        elif lesion_type in malignant_types:
            destination_path = os.path.join(malignant_dest, image_filename)

        if destination_path and os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            copied_count += 1
            if copied_count % 1000 == 0:
                print(f"Copied {copied_count} images so far...")

    print(f"\nDone! A total of {copied_count} images have been organized.")

if __name__ == "__main__":
    organize_dataset()