"""
Script to download real sample images from the PlantVillage dataset on GitHub.
Downloads balanced subsets of real plant leaves for genuine training and testing.
"""

import os
import json
import ssl
import urllib.request
import urllib.parse
import concurrent.futures

CLASSES_TO_DOWNLOAD = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

IMAGES_PER_CLASS = 200
DATASET_BASE = os.path.join("dataset", "PlantVillage")
TEST_SAMPLES_DIR = os.path.join("dataset", "sample_test_images")

# Unverified SSL context for environments with missing Windows certifi stores
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch_file_list(class_name):
    url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{urllib.parse.quote(class_name)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
            files = json.loads(resp.read().decode())
            return [f for f in files if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception as e:
        print(f"Error fetching file list for {class_name}: {e}")
        return []


def download_single_image(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
            content = resp.read()
            with open(target_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def main():
    print("=" * 60)
    print(" Downloading real PlantVillage dataset images from GitHub...")
    print("=" * 60)

    os.makedirs(DATASET_BASE, exist_ok=True)
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for class_name in CLASSES_TO_DOWNLOAD:
            class_dir = os.path.join(DATASET_BASE, class_name)
            os.makedirs(class_dir, exist_ok=True)

            print(f"Fetching manifest for: {class_name}...")
            files = fetch_file_list(class_name)
            if not files:
                continue

            selected = files[:IMAGES_PER_CLASS]
            # Save the first one also to test_samples for quick user testing
            if selected:
                sample_target = os.path.join(TEST_SAMPLES_DIR, f"{class_name}_test.jpg")
                tasks.append(executor.submit(download_single_image, selected[0]["download_url"], sample_target))

            for item in selected:
                target = os.path.join(class_dir, item["name"])
                tasks.append(executor.submit(download_single_image, item["download_url"], target))

        total = len(tasks)
        print(f"\nDownloading {total} real leaf images in parallel...")
        done_count = 0
        for future in concurrent.futures.as_completed(tasks):
            if future.result():
                done_count += 1
            if done_count % 25 == 0 or done_count == total:
                print(f" Progress: {done_count}/{total} images downloaded")

    print("\nDataset setup complete!")
    print(f"Dataset location: {os.path.abspath(DATASET_BASE)}")
    print(f"Sample test images: {os.path.abspath(TEST_SAMPLES_DIR)}")


if __name__ == "__main__":
    main()
