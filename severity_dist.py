import os
import glob
import numpy as np
import cv2
from PIL import Image

DATA_ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"

def get_severity_stats(split):
    les_dir = os.path.join(DATA_ROOT, "masks_lesion", split)
    lf_dir = os.path.join(DATA_ROOT, "masks_leaf", split)

    sevs = []
    # Only images that have BOTH masks are valid for severity
    stems = [os.path.splitext(f)[0] for f in os.listdir(les_dir) if f.endswith(".png")]

    for stem in stems:
        lp = os.path.join(lf_dir, stem + ".png")
        if not os.path.exists(lp): continue

        les = cv2.imread(os.path.join(les_dir, stem + ".png"), 0)
        lf = cv2.imread(lp, 0)

        les_area = np.sum(les > 127)
        lf_area = np.sum(lf > 127)

        if lf_area > 0:
            sevs.append(min(1.0, les_area / lf_area))

    if not sevs: return None

    sevs = np.array(sevs)
    return {
        "median": np.median(sevs),
        "p10": np.percentile(sevs, 10),
        "p90": np.percentile(sevs, 90),
        "count": len(sevs)
    }

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        stats = get_severity_stats(split)
        print(f"{split}: {stats}")
