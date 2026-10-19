import os
import shutil
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

class DataSourcingPipeline:
    """
    Modular Data Sourcing Pipeline for CropGuard.
    Handles acquisition, unification, and versioning of tomato disease datasets.
    """
    def __init__(self, base_dir="CropGuard_Final/data_pipeline"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw_data"
        self.unified_dir = self.base_dir / "unified_dataset"

        # Create directory structure
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.unified_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.unified_dir / "labels").mkdir(parents=True, exist_ok=True)
        (self.unified_dir / "masks").mkdir(parents=True, exist_ok=True)

    def download_dataset(self, url, filename):
        """Downloads a dataset from a URL with basic error handling."""
        target_path = self.raw_dir / filename
        if target_path.exists():
            print(f"Dataset {filename} already exists. Skipping download.")
            return target_path

        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return None
        return target_path

    def extract_archive(self, archive_path):
        """Extracts zip or tar archives to a dedicated folder."""
        extract_to = self.raw_dir / archive_path.stem
        extract_to.mkdir(exist_ok=True)

        print(f"Extracting {archive_path.name} to {extract_to}...")
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            return extract_to
        except Exception as e:
            print(f"Extraction failed for {archive_path.name}: {e}")
            return None

    def unify_format(self, source_folder, class_mapping=None):
        """
        Converts source-specific folder structures into the CropGuard unified format.
        Expected source: folder_per_class/image.jpg
        """
        print(f"Unifying data from {source_folder}...")
        img_count = 0

        # Walk through source folder
        for class_name in os.listdir(source_folder):
            class_path = Path(source_folder) / class_name
            if not class_path.is_dir():
                continue

            # Map class name to index
            class_id = class_mapping.get(class_name, 0) if class_mapping else 0

            for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Copy image
                    src_img = class_path / img_file
                    dst_img = self.unified_dir / "images" / f"{class_name}_{img_file}"
                    shutil.copy(src_img, dst_img)

                    # Create dummy label (Class only) for classification-first datasets
                    # In a full pipeline, we would merge this with mask/box data
                    img_id = dst_img.stem
                    label_path = self.unified_dir / "labels" / f"{img_id}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 0.1 0.1") # Dummy box for unification

                    img_count += 1

        print(f"Successfully unified {img_count} images.")

if __name__ == "__main__":
    # Example usage for developer
    pipeline = DataSourcingPipeline()
    print("Sourcing Pipeline initialized. Ready for dataset integration.")
