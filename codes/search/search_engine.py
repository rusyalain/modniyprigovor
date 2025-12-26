import os
import json
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

class FashionSearchEngine:
    def __init__(self, base_path="/home/borodinro/fashion_attribution/data", device="cpu"):
        self.base_path = Path(base_path)
        self.index_dir = self.base_path / "unified_index"
        self.images_dir = self.base_path / "good_images"
        self.device = device

        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        self.vectors = torch.from_numpy(np.load(self.index_dir / "all_vectors.npy")).to(self.device)
        with open(self.index_dir / "all_metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        null_prompt = "Category: ; Type: ; Formality: ; Gender: ; Color: ; Material: ; Silhouette: ; Season: ; Style: ; Print: ; Fit: ; Length: ; Sleeve: "
        self.null_embedding = self.model.encode(null_prompt, convert_to_tensor=True, device=self.device)

    def parse_json_to_desc(self, item_dict):
        mapping = [
            ("Category", "category_name"), ("Type", "type"), ("Formality", "formality"),
            ("Gender", "gender"), ("Color", "color"), ("Material", "material"),
            ("Silhouette", "silhouette"), ("Season", "season"), ("Style", "style"),
            ("Print", "print"), ("Fit", "fit"), ("Length", "length"), ("Sleeve", "sleeve")
        ]
        parts = [f"{label}: {item_dict.get(key, 'N/A')}" for label, key in mapping]
        return "; ".join(parts)

    def find_best_match(self, query_json_str):
        try:
            data = json.loads(query_json_str)
            items_list = data.get('items', [])
            if isinstance(items_list, dict): items_list = [items_list]
        except Exception as e: return f"JSON Error: {e}"

        results = []
        for item in items_list:
            desc_query = self.parse_json_to_desc(item)
            query_emb = self.model.encode(desc_query, convert_to_tensor=True, device=self.device)
            clean_emb = query_emb - self.null_embedding
            
            similarities = util.cos_sim(clean_emb, self.vectors)[0]
            best_idx = torch.argmax(similarities).item()
            score = similarities[best_idx].item()

            match_data = self.metadata[best_idx]
            results.append({
                "match_score": f"{score*100:.2f}%",
                "set_id": match_data['set_id'],
                "index": match_data['index'],
                "photo": str(self.get_image_path(str(match_data['set_id']), str(match_data['index']))),
                "original_desc": match_data['desc']
            })
        return results

    def get_image_path(self, set_id, index):
        folder = self.images_dir / set_id
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            path = folder / f"{index}{ext}"
            if path.exists(): return path
        return None