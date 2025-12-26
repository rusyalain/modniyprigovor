import os
import json
import torch
import argparse
import warnings
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# --- ПУТИ ---
DATA_DIR = "/home/borodinro/fashion_attribution/data"
JSON_INPUT = os.path.join(DATA_DIR, "full_data.json")
JSON_OUTPUT = os.path.join(DATA_DIR, f"full_data_part_{args.worker_id}.json")
IMAGES_ROOT = os.path.join(DATA_DIR, "good_images")
CATEGORIES_FILE = os.path.join(DATA_DIR, "good_categories.txt")

MODEL_ID = "unsloth/Qwen3-VL-4B-Instruct"
BATCH_SIZE = 8
MAX_TOKENS = 200

# Загрузка справочника категорий
def load_category_map(path):
    cat_map = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2: cat_map[str(parts[0])] = parts[1]
    return cat_map

category_dict = load_category_map(CATEGORIES_FILE)

# Модель
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=512*512)

PROMPT_TPL = """You are a professional stylist. Describe the item in the image using this EXACT template. 
DO NOT drop the field names. Every value must be preceded by its label. 
Output must be in ENGLISH, strictly ONE line. Use 'N/A' if not applicable.

Template:
Type: [value]; Formality: [value]; Gender: [value]; Color: [value]; Material: [value]; Silhouette: [value]; Season: [value]; Style: [value]; Print: [value]; Fit: [value]; Length: [value]; Sleeve: [value]"""

def process_batch(batch_items):
    messages_batch = []
    active_items = []
    for img_path, item_ref in batch_items:
        if os.path.exists(img_path):
            messages_batch.append([{"role": "user", "content": [{"type": "image", "image": f"file://{img_path}"}, {"type": "text", "text": PROMPT_TPL}]}])
            active_items.append(item_ref)
        else:
            item_ref['desc'] = "Error: File not found"

    if not messages_batch: return
    try:
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, repetition_penalty=1.1)
        responses = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for i, response in enumerate(responses):
            item = active_items[i]
            cat_name = category_dict.get(str(item.get('categoryid')), "N/A")
            ai_text = response.strip().replace('\n', ' ').split('Template:')[-1].strip()
            item['desc'] = f"Category: {cat_name}; {ai_text}"
    except Exception as e:
        print(f"Error: {e}")

def main():
    with open(JSON_INPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_todo = []
    for entry in data:
        sid = str(entry.get('set_id'))
        for item in entry.get('items', []):
            if 'desc' not in item or not item['desc']:
                fname = item.get('name', f"{item.get('index')}.jpg")
                img_path = os.path.join(IMAGES_ROOT, sid, fname)
                all_todo.append((img_path, item))

    my_todo = all_todo[args.worker_id::4]
    pbar = tqdm(total=len(my_todo), desc=f"Worker {args.worker_id}")
    for i in range(0, len(my_todo), BATCH_SIZE):
        batch = my_todo[i : i + BATCH_SIZE]
        process_batch(batch)
        pbar.update(len(batch))
        if i % 160 == 0:
            with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()