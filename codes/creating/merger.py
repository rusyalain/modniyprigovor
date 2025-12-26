import json
import os

DATA_DIR = "/home/borodinro/fashion_attribution/data"
BASE_FILE = os.path.join(DATA_DIR, "full_data.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "full_data_FINAL.json")

def merge():
    print("Загрузка базы...")
    with open(BASE_FILE, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    lookup = {}
    for entry in base_data:
        sid = str(entry['set_id'])
        lookup[sid] = {str(item['index']): item for item in entry['items']}

    for i in range(4):
        part_file = os.path.join(DATA_DIR, f"full_data_part_{i}.json")
        if os.path.exists(part_file):
            with open(part_file, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
            for entry in part_data:
                sid = str(entry['set_id'])
                for item in entry['items']:
                    if 'desc' in item and item['desc']:
                        lookup[sid][str(item['index'])]['desc'] = item['desc']
            print(f"Шард {i} добавлен.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)
    print(f"Успех. Файл создан: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge()