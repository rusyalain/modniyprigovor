import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- КОНФИГУРАЦИЯ ---
DATA_DIR = Path("/home/borodinro/fashion_attribution/data")
JSON_FILE = DATA_DIR / "full_data_FINAL.json"
UNIFIED_DIR = DATA_DIR / "unified_index"
CAT_INDEX_DIR = DATA_DIR / "category_index"

VECTORS_FILE = UNIFIED_DIR / "all_vectors.npy"
METADATA_FILE = UNIFIED_DIR / "all_metadata.json"

def count_na(desc):
    if not desc: return 99
    return desc.upper().count("N/A")

def backup(path):
    if path.exists():
        backup_path = path.parent / (path.stem + "_OLD" + path.suffix)
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path.name}")

def cleanup():
    print("--- 1. Filtering main JSON file ---")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    removed_items_count = 0

    for entry in tqdm(data, desc="Filtering JSON"):
        new_items = []
        for item in entry.get('items', []):
            if count_na(item.get('desc', '')) < 6:
                new_items.append(item)
            else:
                removed_items_count += 1
        
        if new_items:
            entry['items'] = new_items
            cleaned_data.append(entry)

    backup(JSON_FILE)
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print("\n--- 2. Filtering Unified Vector Index ---")
    backup(VECTORS_FILE)
    backup(METADATA_FILE)

    vectors = np.load(VECTORS_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    valid_indices = []
    cleaned_metadata = []

    for i, meta in enumerate(tqdm(metadata, desc="Filtering vectors")):
        if count_na(meta.get('desc', '')) < 6:
            valid_indices.append(i)
            cleaned_metadata.append(meta)

    cleaned_vectors = vectors[valid_indices]
    np.save(VECTORS_FILE, cleaned_vectors)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_metadata, f, ensure_ascii=False, indent=2)

    print("\n--- 3. Rebuilding Category Shards ---")
    if CAT_INDEX_DIR.exists():
        old_cat_dir = DATA_DIR / "category_index_OLD"
        if old_cat_dir.exists(): shutil.rmtree(old_cat_dir)
        CAT_INDEX_DIR.rename(old_cat_dir)

    CAT_INDEX_DIR.mkdir()

    cat_groups = {}
    for i, meta in enumerate(cleaned_metadata):
        cid = str(meta['categoryid'])
        if cid not in cat_groups: cat_groups[cid] = []
        cat_groups[cid].append(i)

    for cid, indices in tqdm(cat_groups.items(), desc="Creating shards"):
        cat_path = CAT_INDEX_DIR / f"cat_{cid}"
        cat_path.mkdir()
        cat_vectors = cleaned_vectors[indices]
        np.save(cat_path / "vectors.npy", cat_vectors)
        cat_meta = [cleaned_metadata[idx] for idx in indices]
        with open(cat_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(cat_meta, f, ensure_ascii=False, indent=2)

    print("\nGLOBAL CLEANUP FINISHED!")

if __name__ == "__main__":
    cleanup()