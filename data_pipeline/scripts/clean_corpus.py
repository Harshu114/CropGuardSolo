"""Deduplicate the corpus and remove cross-split leakage.

Why this gates every other improvement: `label_audit_leakage.json` measured 179
dhash groups spanning more than one split (396 images), with **7.45% of val**
involved. `label_audit_result.json` found 2,396 train images (8.09%) sitting in
1,139 duplicate groups, 15 of which carry *conflicting* labels. While that stands,
an improvement on val/test can be leakage rather than learning, so no number can be
trusted.

What this does:
  1. perceptual-hashes (dhash64, the same method as label_audit_dhash.py) every image
  2. groups exact-content duplicates
  3. **drops groups whose members disagree on class** -- contradictory annotation
  4. **keeps one representative per group** and assigns the whole group to ONE
     split (majority vote, ties -> train), so no group can straddle a split again

Outputs, under --out:
    split_assignment.json   stem -> {split, keep, reason, group}
    report.json             before/after counts and what was removed
    images|labels|masks/<split>/   symlinks (cheap; originals untouched)

Nothing in the source corpus is modified or deleted.

    python3 data_pipeline/scripts/clean_corpus.py --root <corpus> --out ./clean_corpus
"""
import argparse
import collections
import json
import os
from datetime import datetime

from PIL import Image

SPLITS = ("train", "val", "test")
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".webp")


def dhash64(path):
    """64-bit difference hash: 9x8 grayscale, compare horizontally adjacent."""
    with Image.open(path) as im:
        g = im.convert("L").resize((9, 8), Image.BILINEAR)
        px = list(g.tobytes())
    bits = 0
    for r in range(8):
        row = px[r * 9:(r + 1) * 9]
        for c in range(8):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def classes_of(label_path):
    try:
        txt = open(label_path, "rb").read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ()
    out = []
    for line in txt.splitlines():
        if line.strip():
            out.append(int(float(line.split()[0])))
    return tuple(sorted(set(out)))


def find_image(img_dir, stem):
    for ext in EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p, ext
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="hash cache json (speeds re-runs)")
    ap.add_argument("--no-materialize", action="store_true")
    args = ap.parse_args()

    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = json.load(open(args.cache))

    # ---- 1/2. hash everything ------------------------------------------------
    info, by_hash = {}, collections.defaultdict(list)
    n = 0
    for split in SPLITS:
        img_dir = os.path.join(args.root, "images", split)
        lbl_dir = os.path.join(args.root, "labels", split)
        if not os.path.isdir(img_dir):
            print(f"  [warn] no images dir for split {split!r}")
            continue
        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in tuple(e.lower() for e in EXTS):
                continue
            key = f"{split}/{stem}"
            if key in cache:
                h = cache[key]
            else:
                try:
                    h = dhash64(os.path.join(img_dir, fn))
                except Exception as e:  # unreadable image
                    print(f"  [warn] unreadable {key}: {e}")
                    continue
                cache[key] = h
            info[key] = {"split": split, "stem": stem, "ext": ext,
                         "cls": classes_of(os.path.join(lbl_dir, stem + ".txt"))}
            by_hash[h].append(key)
            n += 1
    if args.cache:
        json.dump(cache, open(args.cache, "w"))
    print(f"hashed {n} images -> {len(by_hash)} distinct hashes")

    # ---- 3/4. resolve every group -------------------------------------------
    assign, dropped_conflict, dup_removed = {}, 0, 0
    for h, members in by_hash.items():
        cls_sets = {info[m]["cls"] for m in members}
        if len(cls_sets) > 1:
            for m in members:
                assign[m] = {"split": info[m]["split"], "keep": False,
                             "reason": "conflicting_labels", "group": str(h)}
            dropped_conflict += len(members)
            continue
        votes = collections.Counter(info[m]["split"] for m in members)
        # majority split; ties -> train (largest, safest place to absorb a duplicate)
        target = max(votes.items(), key=lambda kv: (kv[1], kv[0] == "train"))[0]
        rep = sorted(members)[0]
        for m in members:
            keep = (m == rep)
            if not keep:
                dup_removed += 1
            assign[m] = {"split": target, "keep": keep,
                         "reason": "representative" if keep else "duplicate",
                         "group": str(h)}

    # ---- report -------------------------------------------------------------
    before = collections.Counter(info[m]["split"] for m in info)
    after = collections.Counter(a["split"] for a in assign.values() if a["keep"])
    moved = sum(1 for m, a in assign.items()
                if a["keep"] and a["split"] != info[m]["split"])
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": os.path.abspath(args.root),
        "images_hashed": len(info),
        "distinct_hashes": len(by_hash),
        "duplicate_images_removed": dup_removed,
        "conflicting_label_images_removed": dropped_conflict,
        "representatives_moved_split": moved,
        "counts_before": dict(before),
        "counts_after": dict(after),
        "method": "dhash64 (9x8 grayscale horizontal difference), same as label_audit_dhash.py",
        "caveat": "exact perceptual content only; near-duplicates differing by crop/"
                  "rotation/colour are NOT caught, so leakage removed here is a lower bound",
    }

    os.makedirs(args.out, exist_ok=True)
    json.dump(assign, open(os.path.join(args.out, "split_assignment.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)

    print(f"\n  duplicates removed      : {dup_removed}")
    print(f"  conflicting labels      : {dropped_conflict} images")
    print(f"  representatives moved   : {moved}")
    print(f"  before: {dict(before)}")
    print(f"  after : {dict(after)}")

    # ---- materialize symlinks ----------------------------------------------
    if not args.no_materialize:
        for split in SPLITS:
            for sub in ("images", "labels", "masks"):
                os.makedirs(os.path.join(args.out, sub, split), exist_ok=True)
        linked = 0
        for key, a in assign.items():
            if not a["keep"]:
                continue
            src_split, stem = info[key]["split"], info[key]["stem"]
            dst_split = a["split"]
            for sub in ("images", "labels", "masks"):
                if sub == "images":
                    src, _ = find_image(os.path.join(args.root, sub, src_split), stem)
                    if src is None:
                        continue
                    name = os.path.basename(src)
                else:
                    ext = ".txt" if sub == "labels" else ".png"
                    src = os.path.join(args.root, sub, src_split, stem + ext)
                    name = stem + ext
                    if not os.path.exists(src):
                        continue
                dst = os.path.join(args.out, sub, dst_split, name)
                if os.path.lexists(dst):
                    continue
                os.symlink(os.path.abspath(src), dst)
                linked += 1
        print(f"  materialized {linked} symlinks under {args.out}")

        yaml = [f"names:"] + [f"- {c}" for c in [
            "healthy", "bacterial_spot", "early_blight", "late_blight", "leaf_miner",
            "leaf_mold", "mosaic_virus", "septoria_leaf_spot", "spider_mites",
            "target_spot", "yellow_leaf_curl_virus"]]
        yaml += ["nc: 11", "train: images/train", "val: images/val", "test: images/test"]
        open(os.path.join(args.out, "data.yaml"), "w").write("\n".join(yaml) + "\n")

    print(f"\nwrote {args.out}/split_assignment.json + report.json")


if __name__ == "__main__":
    main()
