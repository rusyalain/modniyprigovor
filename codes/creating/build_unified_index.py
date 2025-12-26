import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = "/home/borodinro/fashion_attribution/data"
INPUT_FILE = os.path.join(DATA_DIR, "full_data_FINAL.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "unified_index")
VECTORS_FILE = os.path.join(OUTPUT_DIR, "all_vectors.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "all_metadata.json")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model = SentenceTransformer('BAAI/bge-m3', device='cuda')

null_prompt = "Category: ; Type: ; Formality: ; Gender: ; Color: ; Material: ; Silhouette: ; Season: ; Style: ; Print: ; Fit: ; Length: ; Sleeve: "
null_emb = model.encode(null_prompt, convert_to_tensor=True)

def build():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(INPUT_FILE, 'r') as f: data = json.load(f)
    
    texts, meta = [], []
    for entry in data:
        for item in entry['items']:
            desc = item.get('desc')
            if desc and "Category:" in desc:
                texts.append(desc)
                meta.append({"set_id": entry['set_id'], "index": item['index'], "categoryid": item['categoryid'], "desc": desc})

    print(f"Indexing {len(texts)} items...")
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_tensor=True)
    embeddings = embeddings - null_emb
    
    np.save(VECTORS_FILE, embeddings.cpu().numpy())
    with open(METADATA_FILE, 'w') as f: json.dump(meta, f, indent=2)

if __name__ == "__main__":
    build()