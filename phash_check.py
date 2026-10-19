import os
import glob
import cv2
import numpy as np
from PIL import Image

DATA_ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"

def get_phash(img_path):
    img = cv2.imread(img_path, 0)
    if img is None: return None
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    avg = img.mean()
    diff = img > avg
    return diff.flatten()

def find_duplicates():
    splits = ["train", "val", "test"]
    all_images = []
    for split in splits:
        img_dir = os.path.join(DATA_ROOT, "images", split)
        for img_p in glob.glob(os.path.join(img_dir, "*.*")):
            all_images.append((img_p, split))

    hashes = {}
    leaks = 0

    for img_p, split in all_images:
        h = get_phash(img_p)
        if h is None: continue
        h_tuple = tuple(h)
        if h_tuple in hashes:
            other_p, other_split = hashes[h_tuple]
            if split != other_split:
                leaks += 1
                print(f"LEAK: {img_p} ({split}) <-> {other_p} ({other_split})")
        else:
            hashes[h_tuple] = (img_p, split)

    return leaks

if __name__ == "__main__":
    num_leaks = find_duplicates()
    print(f"Total leaking pairs found: {num_leaks}")
