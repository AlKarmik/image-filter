#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE  → --dst
MINORS (tag ∈ bad_tags & score ≥ --thr) → _minors_flagged
Optimized for NVIDIA RTX 3090 GPU on Windows 10
"""

import argparse
import pathlib
import shutil
import logging
import os
import gc
from PIL import Image
from tqdm import tqdm
import torch
from wdtagger import Tagger  # v0.14, swinv2-tagger-v3

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("filter_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Проверка CUDA
if not torch.cuda.is_available():
    logging.error("CUDA недоступен. Убедитесь, что PyTorch с CUDA установлен корректно.")
    exit(1)
logging.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")

# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("--src", required=True)
cli.add_argument("--dst", required=True)
cli.add_argument("--bad", default="F:/minors.txt")
cli.add_argument("--batch", type=int, default=128)  # Подходит для RTX 3090
cli.add_argument("--thr", type=float, default=0.35)
opt = cli.parse_args()

# Проверка файла bad_tags
try:
    bad_tags = {l.strip() for l in open(opt.bad, encoding="utf-8") if l.strip()}
    if not bad_tags:
        logging.error(f"Файл {opt.bad} пуст")
        exit(1)
except FileNotFoundError:
    logging.error(f"Файл {opt.bad} не найден")
    exit(1)
except Exception as e:
    logging.error(f"Ошибка чтения {opt.bad}: {e}")
    exit(1)

# Проверка путей
src = pathlib.Path(opt.src)
dst_ok = pathlib.Path(opt.dst)
dst_bad = src / "_minors_flagged"

if not src.exists() or not src.is_dir():
    logging.error(f"Папка {src} не существует или не является папкой")
    exit(1)
if not dst_ok.parent.exists():
    logging.error(f"Родительская папка для {dst_ok} не существует")
    exit(1)

dst_ok.mkdir(exist_ok=True)
dst_bad.mkdir(exist_ok=True)

# Сбор изображений, пропуск уже перемещённых
img_extensions = {".jpg", ".jpeg", ".png", ".webp"}
processed_files = set(p.name for p in dst_ok.rglob("*") if p.suffix.lower() in img_extensions)
processed_files.update(p.name for p in dst_bad.rglob("*") if p.suffix.lower() in img_extensions)
images = [p for p in src.rglob("*") if p.suffix.lower() in img_extensions and p.name not in processed_files]
logging.info(f"Найдено {len(images)} изображений для обработки")

# Инициализация tagger
try:
    tagger = Tagger()  # Автоматически использует GPU, если CUDA доступен
    # Проверка, что tagger использует GPU
    if not torch.cuda.is_available():
        logging.error("wdtagger не использует GPU")
        exit(1)
except Exception as e:
    logging.error(f"Ошибка инициализации tagger: {e}")
    exit(1)

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def safe_move(src, dst):
    """Перемещение файла с обработкой дубликатов имён"""
    dst_path = dst
    base, ext = os.path.splitext(dst.name)
    counter = 1
    while dst_path.exists():
        dst_path = dst.parent / f"{base}_{counter}{ext}"
        counter += 1
    try:
        shutil.move(src, dst_path)
    except Exception as e:
        logging.error(f"Ошибка перемещения {src} в {dst_path}: {e}")

# Обработка батчей
for batch in tqdm(
    chunks(images, opt.batch),
    total=(len(images) + opt.batch - 1) // opt.batch,
    ncols=80,
    desc="Tagging"
):
    # Загрузка изображений
    pics = []
    valid_paths = []
    for p in batch:
        try:
            img = Image.open(p).convert("RGB")
            pics.append(img)
            valid_paths.append(p)
        except Exception as e:
            logging.warning(f"Ошибка открытия {p}: {e}")
            continue

    if not pics:
        continue

    # Тегирование на GPU
    try:
        with torch.no_grad():  # Экономия VRAM
            results = tagger.tag(pics)  # list[Result]
        logging.info(f"Tagged {len(pics)} images in batch")
    except Exception as e:
        logging.error(f"Ошибка тегирования батча: {e}")
        continue

    # Обработка результатов
    for img_path, res in zip(valid_paths, results):
        try:
            tags = {**res.general_tag_data, **res.character_tag_data}
            is_minor = any(
                tag in bad_tags and score >= opt.thr
                for tag, score in tags.items()
            )
            dest = dst_bad if is_minor else dst_ok
            safe_move(img_path, dest / img_path.name)
            if is_minor:
                flagged_tags = [
                    (tag, score) for tag, score in tags.items()
                    if tag in bad_tags and score >= opt.thr
                ]
                logging.info(f"Flagged {img_path.name}: {flagged_tags}")
            else:
                logging.info(f"Safe {img_path.name}")
        except Exception as e:
            logging.error(f"Ошибка обработки {img_path}: {e}")

    # Очистка памяти
    pics = None
    torch.cuda.empty_cache()
    gc.collect()

# Подсчёт результатов
ok_count = sum(1 for p in dst_ok.rglob("*") if p.suffix.lower() in img_extensions)
bad_count = sum(1 for p in dst_bad.rglob("*") if p.suffix.lower() in img_extensions)
logging.info(f"✓ SAFE: {ok_count} | FLAGGED: {bad_count}")
print(f"✓ SAFE: {ok_count} | FLAGGED: {bad_count}")
