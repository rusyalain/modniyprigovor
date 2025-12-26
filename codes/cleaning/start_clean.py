#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
copy_good_images_strict.py

Предполагает:
 - папки в images_root имеют имена, которые точно равны set_id (только цифры), напр.: 119704139
 - файлы внутри имеют имена, которые точно состоят из цифр (stem), напр.: 0.jpg, 12.png

По умолчанию dry-run (только печатает). Для реального копирования добавьте --do-copy.

ДОПОЛНИТЕЛЬНАЯ ФУНКЦИОНАЛЬНОСТЬ:
Если папка содержит хотя бы один хороший файл (с categoryid из good_categories),
то в нее обязательно копируется файл с именем "0" (независимо от его categoryid),
если такой файл существует в исходной папке.
"""

import argparse
import json
import re
from pathlib import Path
import shutil
from typing import Dict, Set, List, Tuple, Optional

# ---------------------------------------
def load_sets_mapping_from_file(json_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Возвращает mapping: set_id_str -> { index_int -> categoryid_int_or_None }
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    mapping: Dict[str, Dict[int, int]] = {}
    for entry in 
        # ожидаем, что set_id представлен в поле 'set_id' или 'id'
        set_id = None
        for key in ('set_id', 'setId', 'id', 'name'):
            if key in entry and entry[key] not in (None, ''):
                set_id = str(entry[key])
                break
        if not set_id:
            continue

        items = entry.get('items') or []
        idx_map: Dict[int, int] = {}
        for it in items:
            # index -> integer
            try:
                idx = int(it.get('index'))
            except Exception:
                continue
            cat = it.get('categoryid')
            try:
                cat_int = int(cat) if cat is not None else None
            except Exception:
                cat_int = None
            idx_map[idx] = cat_int
        mapping[set_id] = idx_map
    return mapping

def load_good_categories(path: Path) -> Set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Good categories file not found: {path}")
    cats = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'\s*([0-9]+)', line)
            if m:
                cats.add(int(m.group(1)))
    return cats

# ---------------------------------------
def find_zero_file(folder: Path, allowed_exts: Set[str]) -> Optional[Path]:
    """
    Ищет файл с именем '0' с любым из разрешенных расширений в указанной папке.
    Возвращает путь к файлу или None, если не найден.
    """
    for ext in allowed_exts:
        candidate = folder / f"0.{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None

def main():
    p = argparse.ArgumentParser(description="Strict copy of good images (folder name == set_id, filenames digits only)")
    p.add_argument('--test-json', default=r"D:\project\test_no_dup.json", help='Path to test JSON')
    p.add_argument('--train-json', default=r"D:\project\train_no_dup.json", help='Path to train JSON')
    p.add_argument('--images-root', default=r"D:\project\images", help='Root folder with image subfolders (each named exactly set_id)')
    p.add_argument('--good-cats', default=r"D:\project\good_categories.txt", help='Good categories file (lines start with categoryid number)')
    p.add_argument('--out-root', default=r"D:\project\good_images", help='Output root for copied good images')
    p.add_argument('--do-copy', action='store_true', help='Actually copy files (default = dry-run)')
    p.add_argument('--overwrite', action='store_true', help='If copying, overwrite existing files at destination')
    p.add_argument('--exts', default='jpg,jpeg,png', help='Allowed extensions (comma-separated)')
    p.add_argument('--verbose', action='store_true', help='Verbose output')
    p.add_argument('--log-file', default=None, help='CSV log file to record copied (or would-copy) operations')
    args = p.parse_args()
    args.do_copy = True
    args.overwrite = True
    args.verbose = True
    print("DEBUG argv do_copy =", args.do_copy)
    print("DEBUG images_root =", args.images_root)
    print("DEBUG out_root =", args.out_root)

    images_root = Path(args.images_root)
    out_root = Path(args.out_root)
    if not images_root.exists() or not images_root.is_dir():
        raise SystemExit(f"Images root not found or not a directory: {images_root}")

    allowed_exts = set(x.lower().lstrip('.') for x in args.exts.split(','))

    # load and merge JSONs
    sets_map: Dict[str, Dict[int, int]] = {}
    for jp in (Path(args.test_json), Path(args.train_json)):
        if not jp.exists():
            print(f"[WARN] json missing, skip: {jp}")
            continue
        m = load_sets_mapping_from_file(jp)
        for k, v in m.items():
            if k in sets_map:
                sets_map[k].update(v)
            else:
                sets_map[k] = v

    good_cats = load_good_categories(Path(args.good_cats))

    if args.verbose:
        print(f"Loaded sets: {len(sets_map)}")
        print(f"Good categories: {len(good_cats)}")
        print(f"Dry-run: {not args.do_copy}; out-root: {out_root}; allowed exts: {sorted(allowed_exts)}")

    # prepare logging if requested
    log_fh = None
    log_writer = None
    if args.log_file:
        import csv
        log_fh = open(args.log_file, 'w', encoding='utf-8', newline='')
        log_writer = csv.writer(log_fh)
        log_writer.writerow(['action', 'src', 'dst', 'set_id', 'index', 'categoryid', 'mandatory_zero'])

    total_sets = 0
    total_candidates = 0
    total_copied = 0
    total_mandatory_zero_added = 0

    # Iterate only folders with numeric names (strict)
    for sub in images_root.iterdir():
        if not sub.is_dir():
            continue
        if not sub.name.isdigit():
            # skip non-numeric folder names (strict mode)
            if args.verbose:
                print(f"[SKIP] non-numeric folder name: {sub.name}")
            continue

        set_id = sub.name  # exact match expected
        mapping = sets_map.get(set_id)
        if mapping is None:
            # also try without leading zeros if present (defensive)
            alt = set_id.lstrip('0')
            mapping = sets_map.get(alt)
        if mapping is None:
            if args.verbose:
                print(f"[SKIP] set_id not found in JSON: {set_id}")
            continue

        # collect good files in this folder
        good_files: List[Tuple[Path, int, Optional[int]]] = []
        for f in sub.iterdir():
            if not f.is_file():
                continue
            ext = f.suffix.lower().lstrip('.')
            if ext not in allowed_exts:
                continue
            stem = f.stem
            if not stem.isdigit():
                # strict -> skip
                if args.verbose:
                    print(f"  [SKIP] filename not numeric stem: {f.name}")
                continue
            idx = int(stem)
            cat = mapping.get(idx)
            if cat is None:
                continue
            if cat not in good_cats:
                continue
            good_files.append((f, idx, cat))

        # Skip if no good files found (not a "good" set)
        if not good_files:
            if args.verbose:
                print(f"[OK] set {set_id}: no good files found")
            continue

        # === ДОПОЛНИТЕЛЬНАЯ ФУНКЦИОНАЛЬНОСТЬ ===
        # Если есть хорошие файлы, обязательно добавляем файл "0" если он существует
        zero_file_added = False
        zero_file = find_zero_file(sub, allowed_exts)
        if zero_file:
            # Проверяем, не добавлен ли файл "0" уже как хороший файл
            zero_already_included = any(
                src.name == zero_file.name for src, _, _ in good_files
            )
            
            if not zero_already_included:
                # Получаем index и category для файла "0" (даже если они не хорошие)
                zero_idx = 0
                zero_cat = mapping.get(zero_idx) if mapping else None
                
                # Добавляем файл "0" в список для копирования
                good_files.append((zero_file, zero_idx, zero_cat))
                zero_file_added = True
                total_mandatory_zero_added += 1
                if args.verbose:
                    print(f"  [MANDATORY] added zero file: {zero_file.name} (categoryid={zero_cat})")

                # Запись в лог для mandatory zero файла
                if log_writer:
                    dst_dir = out_root / set_id
                    dst = dst_dir / zero_file.name
                    log_writer.writerow([
                        'mandatory_zero_added',
                        str(zero_file),
                        str(dst),
                        set_id,
                        zero_idx,
                        zero_cat if zero_cat is not None else 'None',
                        True
                    ])
        # === КОНЕЦ ДОПОЛНИТЕЛЬНОЙ ФУНКЦИОНАЛЬНОСТИ ===

        # have >=1 good file (including possibly mandatory zero) -> ensure destination folder
        total_sets += 1
        dst_dir = out_root / set_id
        if args.do_copy:
            dst_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"[SET] {set_id}: {len(good_files)} files to copy (including mandatory zero={zero_file_added}) -> {dst_dir} (do_copy={args.do_copy})")

        for src, idx, cat in good_files:
            dst = dst_dir / src.name
            total_candidates += 1
            
            # Определяем, является ли этот файл mandatory zero файлом
            is_mandatory_zero = (zero_file_added and src.name.startswith('0.'))
            
            if args.do_copy:
                if dst.exists() and not args.overwrite:
                    if args.verbose:
                        print(f"  [SKIP] exists (no-overwrite): {dst}")
                    if log_writer:
                        log_writer.writerow(['skipped_exists', str(src), str(dst), set_id, idx, cat if cat is not None else 'None', is_mandatory_zero])
                    continue
                try:
                    shutil.copy2(str(src), str(dst))
                    total_copied += 1
                    if args.verbose:
                        action_desc = "mandatory zero" if is_mandatory_zero else "good"
                        print(f"  copied ({action_desc}): {src.name} -> {dst}")
                    if log_writer:
                        log_writer.writerow(['copied', str(src), str(dst), set_id, idx, cat if cat is not None else 'None', is_mandatory_zero])
                except Exception as e:
                    print(f"[ERROR] failed to copy {src} -> {dst}: {e}")
                    if log_writer:
                        log_writer.writerow(['error', str(src), str(dst), set_id, idx, cat if cat is not None else 'None', is_mandatory_zero])
            else:
                # dry-run
                action_desc = "mandatory zero" if is_mandatory_zero else "good"
                if args.verbose:
                    print(f"  would copy ({action_desc}): {src} -> {dst}")
                if log_writer:
                    log_writer.writerow(['would_copy', str(src), str(dst), set_id, idx, cat if cat is not None else 'None', is_mandatory_zero])

    if log_fh:
        log_fh.close()

    print("=== Summary ===")
    print(f"Sets with >=1 good file: {total_sets}")
    print(f"Total candidate files found: {total_candidates}")
    print(f"Total mandatory zero files added: {total_mandatory_zero_added}")
    print(f"Total files actually copied: {total_copied} (do_copy={args.do_copy})")
    if args.do_copy:
        print(f"Successfully processed {total_sets} good sets with mandatory zero files where applicable")

if __name__ == '__main__':
    main()