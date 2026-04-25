#!/usr/bin/env python3
import argparse
import hashlib
import imghdr
import io
import json
import logging
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Dict, Any
from urllib.parse import parse_qs, urljoin, urlparse

import face_recognition
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, ImageEnhance
from zoneinfo import ZoneInfo
from collections import defaultdict

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except Exception:
    pytesseract = None
    TesseractOutput = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
FILENAME_PREFIX = "yahoo_rt"
FACE_MODEL = "hog"
FACE_UPSAMPLE = 1
OCR_LANG = "jpn+eng"
POSTER_TEXT_RE = re.compile(
    r"(open|start|live|ticket|door|adv|present|presents|schedule|vol\.?|women|"
    r"会場|開場|開演|前売|当日|料金|出演|チケット|女性限定|特典会|入場)"
)
DATE_TIME_RE = re.compile(
    r"(\d{1,2}[:：]\d{2}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}|"
    r"\d{1,2}月\d{1,2}日)"
)
DEFAULT_URLS = [
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_noka&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_idol&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_ena&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_ren&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_roko&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_ami&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_kii&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_yuki&aq=-1&ei=UTF-8&mtype=image&rkf=1",
    "https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_iroha&aq=-1&ei=UTF-8&mtype=image&rkf=1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Yahoo!リアルタイム検索から画像を取得し、既知の顔にマッチするものだけ残します。"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=DEFAULT_URLS,
        help="Yahoo!リアルタイム検索などのページURL。",
    )
    parser.add_argument(
        "--html-file",
        default="",
        help="保存済みHTMLファイルを解析してダウンロードする場合のパス。",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="--html-file使用時の基点URL。",
    )
    parser.add_argument(
        "--usage-file",
        default="state/usage.json",
        help="Path to usage metadata that stores first_seen timestamps.",
    )
    parser.add_argument(
        "--out-dir",
        default="images",
        help="保存先ディレクトリ。",
    )
    parser.add_argument(
        "--reference-dir",
        default="MEGAFON_noka",
        help="参照用の顔画像ディレクトリ（基準顔）。",
    )
    parser.add_argument(
        "--negative-dir",
        default="MEGAFON_other",
        help="除外したい顔画像ディレクトリ（NG顔）。",
    )
    parser.add_argument(
        "--clean-training-dirs",
        action="store_true",
        help="reference/negativeディレクトリから顔が取れない画像を削除して整理する。",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=True,
        help="学習キャッシュ(.encodings_cache.pkl)を優先して使用する。",
    )
    parser.add_argument(
        "--no-cache-only",
        dest="cache_only",
        action="store_false",
        help="キャッシュが古い場合に再計算を許可する。",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=2,
        help="実行日からさかのぼって何日までの投稿を対象にするか。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.50, # 変更: デフォルトを0.45から0.50へ緩和
        help="顔マッチの厳しさ（小さいほど厳密。0.6が標準、0.5推奨）。",
    )
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=0.40, # 変更: デフォルトを少し緩和
        help="NG顔とみなす距離（これ以下だと弾く）。",
    )
    parser.add_argument(
        "--negative-margin",
        type=float,
        default=0.05, # 変更: マージンを少し広げる
        help="NG顔と紛らわしい場合のマージン。",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=5, # 変更: 精度向上のためデフォルトを3から5へ
        help="顔エンコード時のジッター回数（増やすと精度↑）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="顔処理に使うプロセス数。",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=2, # 変更: 写り込みを考慮してデフォルトを2から3へ
        help="1枚あたりの許容顔数。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTPタイムアウト秒数。",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=1, # 変更: デフォルトを0から1へ（検出率向上）
        help="顔検出のアップサンプル回数。",
    )
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="顔検出モデル（cnnの方が精度高いがGPU推奨）。",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="ログファイルパス。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="詳細ログを出力する。",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTPリクエストのUser-Agent。",
    )
    parser.add_argument(
        "--block-list",
        default="state/block.json",
        help="保存をスキップする画像のハッシュ/ファイル名を列挙したJSONファイル。",
    )
    parser.add_argument(
        "--no-ocr",
        dest="ocr_enabled",
        action="store_false",
        help="Disable OCR-based event-poster rejection.",
    )
    parser.set_defaults(ocr_enabled=True)
    parser.add_argument(
        "--ocr-min-text-items",
        type=int,
        default=8,
        help="Minimum OCR token count before poster heuristics engage.",
    )
    parser.add_argument(
        "--ocr-min-text-chars",
        type=int,
        default=24,
        help="Minimum OCR character count for poster rejection.",
    )
    parser.add_argument(
        "--ocr-min-text-area",
        type=float,
        default=0.12,
        help="Minimum OCR box area ratio for poster rejection.",
    )
    return parser.parse_args()


def setup_logging(log_file: str, debug: bool) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def compute_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_usage_state(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Usage file is empty or invalid JSON; resetting: %s", path)
        return {}
    if isinstance(raw, dict):
        return raw
    logging.warning("Usage file is malformed: %s", path)
    return {}


def save_usage_state(path: Path, usage: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def normalize_usage_entry(raw: object) -> Dict[str, float]:
    entry = {"count": 0, "last_used": 0.0, "first_seen": 0.0}
    if isinstance(raw, dict):
        count = raw.get("count", 0)
        if isinstance(count, (int, float)):
            entry["count"] = int(count)
        for key in ("last_used", "first_seen"):
            value = raw.get(key, entry[key])
            if isinstance(value, (int, float)):
                entry[key] = float(value)
    elif isinstance(raw, (int, float)):
        entry["count"] = int(raw)
    return entry


def ensure_usage_entry(
    usage: Dict[str, Dict[str, float]],
    image_name: str,
    first_seen_ts: float,
) -> bool:
    entry = normalize_usage_entry(usage.get(image_name))
    changed = usage.get(image_name) != entry
    if entry["first_seen"] <= 0:
        entry["first_seen"] = first_seen_ts
        changed = True
    if changed:
        usage[image_name] = entry
    return changed


def load_block_list(path: Path) -> Set[str]:
    """
    JSONファイルからブロック対象のダイジェスト接頭辞を読み込む。
    - リスト形式、もしくは {block: [...]} / {digests: [...]} などの配列を想定。
    - 要素は sha256 の先頭数文字、あるいは yahoo_rt_xxx.jpg のファイル名でもOK。
    """
    blocked: Set[str] = set()
    if not path:
        return blocked
    try:
        raw_text = path.read_text(encoding="utf-8")
        if not raw_text.strip():
            return blocked
        raw = json.loads(raw_text)
    except FileNotFoundError:
        return blocked
    except Exception:
        logging.warning("Failed to load block list: %s", path)
        raw = re.findall(
            rf"{re.escape(FILENAME_PREFIX)}_[0-9a-fA-F]+|[0-9a-fA-F]{{8,64}}",
            path.read_text(encoding="utf-8", errors="ignore"),
        )

    if isinstance(raw, dict):
        for key in ("block", "digests", "files", "items"):
            if key in raw and isinstance(raw[key], (list, tuple, set)):
                raw = raw[key]
                break

    if not isinstance(raw, (list, tuple, set)):
        return blocked

    for entry in raw:
        if entry is None:
            continue
        s = str(entry).strip().lower()
        if s.startswith(FILENAME_PREFIX + "_"):
            s = s.split("_", 1)[1]
        if "." in s:
            s = s.split(".", 1)[0]
        s = re.sub(r"[^0-9a-f]", "", s)
        if s:
            blocked.add(s)
    return blocked


def deduplicate_images(images_dir: Path) -> None:
    files = list_images(images_dir)
    if not files:
        return

    info_list = []
    for path in files:
        try:
            data = path.read_bytes()
            digest = compute_digest(data)
            info_list.append({"path": path, "digest": digest})
        except Exception:
            logging.warning("画像の読み込みに失敗しました: %s", path)

    to_delete: Set[Path] = set()
    by_digest = {}
    for info in info_list:
        digest = info["digest"]
        if digest not in by_digest:
            by_digest[digest] = info
            continue
        to_delete.add(info["path"])

    if to_delete:
        for path in to_delete:
            try:
                path.unlink()
                logging.info("重複画像を削除しました: %s", path)
            except Exception:
                logging.warning("重複画像の削除に失敗しました: %s", path)


def perceptual_hash(path: Path) -> Optional[int]:
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((8, 8), Image.LANCZOS)
            pixels = list(img.getdata())
    except Exception:
        logging.warning("p-hash作成に失敗: %s", path)
        return None
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def remove_near_duplicates(
    images_dir: Path,
    new_files: List[Path],
    phash_threshold: int = 3,
) -> int:
    if not new_files:
        return 0

    existing_files = [p for p in list_images(images_dir) if p not in new_files]
    existing_hashes = {}
    for p in existing_files:
        h = perceptual_hash(p)
        if h is not None:
            existing_hashes[p] = h

    to_delete: List[Path] = []
    for p in new_files:
        h_new = perceptual_hash(p)
        if h_new is None:
            continue
        for h_old in existing_hashes.values():
            if hamming_distance(h_new, h_old) <= phash_threshold:
                to_delete.append(p)
                logging.info("Remove near-duplicate (p-hash): %s", p)
                break

    for p in to_delete:
        try:
            p.unlink()
        except Exception:
            logging.warning("近似重複の削除に失敗: %s", p)
    return len(to_delete)


TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def parse_time_text(text: str, now: datetime) -> Optional[int]:
    text = text.strip()
    m = re.match(r"(\d+)\s*分前", text)
    if m:
        minutes = int(m.group(1))
        dt = now - timedelta(minutes=minutes)
        return int(dt.timestamp())
    m = re.match(r"(\d+)\s*時間前", text)
    if m:
        hours = int(m.group(1))
        dt = now - timedelta(hours=hours)
        return int(dt.timestamp())
    m = re.match(r"(昨日|今日)\s*(\d{1,2}):(\d{2})", text)
    if m:
        base = now.date()
        if m.group(1) == "昨日":
            base = base - timedelta(days=1)
        hour, minute = int(m.group(2)), int(m.group(3))
        dt = datetime.combine(base, datetime.min.time(), tzinfo=now.tzinfo).replace(
            hour=hour, minute=minute
        )
        return int(dt.timestamp())
    m = re.match(r"(\d{1,2})月(\d{1,2})日.*?(\d{1,2}):(\d{2})", text)
    if m:
        month, day, hour, minute = map(int, m.groups())
        year = now.year
        dt = datetime(year, month, day, hour, minute, tzinfo=now.tzinfo)
        if dt > now + timedelta(days=1):
            dt = dt.replace(year=year - 1)
        return int(dt.timestamp())
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(\d{1,2}):(\d{2})", text)
    if m:
        year, month, day, hour, minute = map(int, m.groups())
        dt = datetime(year, month, day, hour, minute, tzinfo=now.tzinfo)
        return int(dt.timestamp())
    return None


def digests_for_existing(images_dir: Path) -> Set[str]:
    digests: Set[str] = set()
    for path in list_images(images_dir):
        try:
            with open(path, "rb") as f:
                digests.add(compute_digest(f.read()))
        except Exception:
            logging.warning("Could not hash %s; skipping.", path)
    return digests


def load_known_encodings(reference_dir: Path, num_jitters: int, cache_only: bool) -> List:
    ref_paths = list_images(reference_dir)
    file_hashes = {}
    for path in ref_paths:
        try:
            data = path.read_bytes()
            file_hashes[str(path)] = compute_digest(data)
        except Exception:
            logging.warning("Failed to hash reference image %s; skipping.", path)

    cache_path = reference_dir / ".encodings_cache.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                cached_enc = cached.get("encodings") or []
                if cached_enc and (
                    cache_only or cached.get("file_hashes") == file_hashes
                ):
                    logging.info(
                        "Loaded cached reference encodings (%d).", len(cached_enc)
                    )
                    return cached_enc
        except Exception:
            logging.debug("Failed to load encoding cache; rebuilding.", exc_info=True)

    encodings = []
    requested = int(getattr(load_known_encodings, "_workers", 2) or 2)
    cpu = os.cpu_count() or 1
    if requested <= 0:
        requested = cpu
    max_workers = max(1, min(cpu, requested))
    if len(ref_paths) == 1:
        max_workers = 1
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_encode_faces_from_path, str(p), num_jitters) for p in ref_paths]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                _, file_enc = fut.result()
                if file_enc:
                    encodings.extend(file_enc)
            except Exception:
                logging.warning("Failed to process reference image; skipping.", exc_info=True)
            if done % 50 == 0 or done == len(futures):
                logging.info("Reference encoding progress: %d/%d", done, len(futures))
    if not encodings:
        raise RuntimeError("No face encodings found in reference images.")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"file_hashes": file_hashes, "encodings": encodings}, f)
        logging.info(
            "Loaded %d reference encodings and cached to %s.", len(encodings), cache_path
        )
    except Exception:
        logging.debug("Failed to write encoding cache.", exc_info=True)
        logging.info("Loaded %d reference face encodings.", len(encodings))
    return encodings


def load_negative_encodings(negative_dir: Path, num_jitters: int, cache_only: bool) -> List:
    if not negative_dir or not negative_dir.exists():
        return []
    neg_paths = list_images(negative_dir)
    if not neg_paths:
        return []

    file_hashes = {}
    for path in neg_paths:
        try:
            data = path.read_bytes()
            file_hashes[str(path)] = compute_digest(data)
        except Exception:
            logging.warning("Failed to hash negative image %s; skipping.", path)

    cache_path = negative_dir / ".encodings_cache.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                cached_enc = cached.get("encodings") or []
                if cached_enc and (
                    cache_only or cached.get("file_hashes") == file_hashes
                ):
                    logging.info(
                        "Loaded cached negative encodings (%d).", len(cached_enc)
                    )
                    return cached_enc
        except Exception:
            logging.debug("Failed to load negative encoding cache; rebuilding.", exc_info=True)

    encodings = []
    requested = int(getattr(load_negative_encodings, "_workers", 2) or 2)
    cpu = os.cpu_count() or 1
    if requested <= 0:
        requested = cpu
    max_workers = max(1, min(cpu, requested))
    if len(neg_paths) == 1:
        max_workers = 1
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_encode_faces_from_path_with_count, str(p), num_jitters)
            for p in neg_paths
        ]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                _, file_enc, _ = fut.result()
                if file_enc:
                    encodings.extend(file_enc)
            except Exception:
                logging.warning("Failed to process negative image; skipping.", exc_info=True)
            if done % 100 == 0 or done == len(futures):
                logging.info("Negative encoding progress: %d/%d", done, len(futures))

    if not encodings:
        logging.info("No negative face encodings found in %s.", negative_dir)
        return []

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"file_hashes": file_hashes, "encodings": encodings}, f)
        logging.info(
            "Loaded %d negative encodings and cached to %s.", len(encodings), cache_path
        )
    except Exception:
        logging.debug("Failed to write negative encoding cache.", exc_info=True)
        logging.info("Loaded %d negative face encodings.", len(encodings))
    return encodings


def _clean_face_dir(dir_path: Path, num_jitters: int, label: str) -> int:
    if not dir_path.exists():
        return 0
    files = list_images(dir_path)
    if not files:
        return 0

    requested = int(getattr(_clean_face_dir, "_workers", 2) or 2)
    cpu = os.cpu_count() or 1
    if requested <= 0:
        requested = cpu
    max_workers = max(1, min(cpu, requested))
    if len(files) == 1:
        max_workers = 1

    removed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_has_any_face_in_path, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            path = futures[fut]
            ok = False
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if not ok:
                try:
                    path.unlink()
                    removed += 1
                    logging.info("[%s] Removed unusable face image: %s", label, path)
                except Exception:
                    logging.warning("[%s] Failed to remove bad image: %s", label, path)
            if done % 200 == 0 or done == len(files):
                logging.info("[%s] Clean progress: %d/%d removed=%d", label, done, len(files), removed)
    if removed:
        logging.info("[%s] Cleaned %s: removed=%d", label, dir_path, removed)
    return removed


def _encode_faces_from_path(path_str: str, num_jitters: int):
    path = Path(path_str)
    data = path.read_bytes()
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(
        image, model=FACE_MODEL, number_of_times_to_upsample=FACE_UPSAMPLE
    )
    encodings = face_recognition.face_encodings(image, locations, num_jitters=num_jitters)
    return path_str, encodings


def _encode_faces_from_path_with_count(path_str: str, num_jitters: int):
    path = Path(path_str)
    data = path.read_bytes()
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(
        image, model=FACE_MODEL, number_of_times_to_upsample=FACE_UPSAMPLE
    )
    encodings = face_recognition.face_encodings(image, locations, num_jitters=num_jitters)
    return path_str, encodings, len(locations)


def _has_any_face_in_path(path_str: str) -> bool:
    path = Path(path_str)
    data = path.read_bytes()
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(
        image, model=FACE_MODEL, number_of_times_to_upsample=FACE_UPSAMPLE
    )
    return bool(locations)


def fetch_search_page(
    url: str, cursor: Optional[str], timeout: float, user_agent: str
) -> str:
    params = {"cursor": cursor} if cursor else None
    resp = requests.get(
        url, params=params, headers={"User-Agent": user_agent}, timeout=timeout
    )
    resp.raise_for_status()
    return resp.text


def extract_image_urls(
    html: str, base_url: str
) -> Tuple[List[Tuple[str, Optional[int]]], Optional[str]]:
    def _collect_media_from_container(container) -> List[str]:
        found: List[str] = []
        if not container:
            return found
        for img in container.find_all("img"):
            src = img.get("src")
            if src:
                found.append(src)
        for div in container.find_all(style=True):
            style = div.get("style", "")
            if "background-image" not in style:
                continue
            bg_url = _extract_background_url(style)
            if bg_url:
                found.append(bg_url)
        return found

    def _extract_background_url(style_value: str) -> Optional[str]:
        match = re.search(r"background-image:\s*url\(([^)]+)\)", style_value)
        if not match:
            return None
        raw = match.group(1).strip().strip("\"'")
        return raw or None

    def media_urls_from_entry(entry: dict) -> List[Tuple[str, Optional[int]]]:
        urls: List[Tuple[str, Optional[int]]] = []
        created_at = entry.get("createdAt")

        def add_from_media(media: dict) -> None:
            if not isinstance(media, dict):
                return
            item = media.get("item") or {}
            for source in (media, item):
                for key in ("mediaUrl", "metaImageUrl", "thumbnailImageUrl"):
                    val = source.get(key)
                    if isinstance(val, str):
                        urls.append((val, created_at))

        for media in entry.get("media") or []:
            add_from_media(media)

        quoted = entry.get("quotedTweet") or {}
        for media in quoted.get("media") or []:
            add_from_media(media)

        return urls

    soup = BeautifulSoup(html, "html.parser")
    urls: List[Tuple[str, Optional[int]]] = []
    now_jst = datetime.now(TOKYO_TZ)

    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            data = json.loads(next_data.string)
            page_data = data.get("props", {}).get("pageProps", {}).get("pageData", {})
            entries = page_data.get("timeline", {}).get("entry") or []
            if isinstance(page_data.get("bestTweet"), dict):
                entries.append(page_data["bestTweet"])
            for entry in entries:
                if isinstance(entry, dict):
                    urls.extend(media_urls_from_entry(entry))
            if urls:
                return urls, None
        except Exception:
            logging.warning(
                "Failed to parse Yahoo! realtime search payload; falling back to HTML.",
                exc_info=True,
            )

    tweet_containers = soup.find_all(
        "div", class_=lambda c: c and "Tweet_TweetContainer" in c
    )
    if tweet_containers:
        for container in tweet_containers:
            ts: Optional[int] = None
            time_tag = container.find("time")
            if time_tag:
                ts = parse_time_text(time_tag.get_text(strip=True), now_jst)
            for src in _collect_media_from_container(container):
                urls.append((urljoin(base_url, src), ts))
    else:
        time_values: List[Optional[int]] = []
        for t in soup.find_all("time"):
            parsed = parse_time_text(t.get_text(strip=True), now_jst)
            if parsed:
                time_values.append(parsed)
        time_idx = 0

        for a in soup.find_all("a", class_=lambda c: c and "still-image" in c):
            href = a.get("href")
            if href:
                ts = time_values[time_idx] if time_idx < len(time_values) else None
                time_idx += 1
                urls.append((urljoin(base_url, href), ts))

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            if img.get("data-test") == "image" or (
                img.parent and "Tweet_imageContainer" in (img.parent.get("class") or [])
            ):
                ts = time_values[time_idx] if time_idx < len(time_values) else None
                time_idx += 1
                urls.append((urljoin(base_url, src), ts))
            elif "/pic/" in src:
                ts = time_values[time_idx] if time_idx < len(time_values) else None
                time_idx += 1
                urls.append((urljoin(base_url, src), ts))

        for div in soup.find_all(style=True):
            style = div.get("style", "")
            if "background-image" not in style:
                continue
            bg_url = _extract_background_url(style)
            if not bg_url:
                continue
            ts = time_values[time_idx] if time_idx < len(time_values) else None
            time_idx += 1
            urls.append((urljoin(base_url, bg_url), ts))

    next_cursor = None
    more_link = soup.find("a", href=lambda h: h and "cursor=" in h)
    if more_link:
        href = more_link.get("href", "")
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "cursor" in qs and qs["cursor"]:
            next_cursor = qs["cursor"][0]

    bt_container = soup.find("div", id="bt")
    if bt_container:
        bt_ts = None
        bt_time = bt_container.find("time")
        if bt_time:
            bt_ts = parse_time_text(bt_time.get_text(strip=True), now_jst)
        for src in _collect_media_from_container(bt_container):
            urls.append((urljoin(base_url, src), bt_ts))

    return urls, next_cursor


def determine_extension(data: bytes) -> Optional[str]:
    kind = imghdr.what(None, h=data)
    if kind == "jpeg":
        return ".jpg"
    if kind in ("png", "gif", "webp"):
        return f".{kind}"
    return None


def _prepare_ocr_image(pil_img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(pil_img.convert("L"))
    max_dim = max(img.size)
    if max_dim < 1400:
        scale = 1400.0 / max_dim
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.LANCZOS,
        )
    return img


def detect_poster_text(
    pil_img: Image.Image,
    min_text_items: int,
    min_text_chars: int,
    min_text_area: float,
) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {
        "available": bool(pytesseract and TesseractOutput),
        "token_count": 0,
        "char_count": 0,
        "text_area_ratio": 0.0,
        "keyword_hits": 0,
        "datetime_hits": 0,
    }
    if not pytesseract or not TesseractOutput:
        return False, details

    img = _prepare_ocr_image(pil_img)
    try:
        data = pytesseract.image_to_data(
            img,
            lang=OCR_LANG,
            config="--oem 3 --psm 6",
            output_type=TesseractOutput.DICT,
        )
    except Exception:
        try:
            data = pytesseract.image_to_data(
                img,
                lang="eng",
                config="--oem 3 --psm 6",
                output_type=TesseractOutput.DICT,
            )
        except Exception:
            return False, details

    tokens: List[str] = []
    text_area = 0
    width = max(1, img.size[0])
    height = max(1, img.size[1])
    confidences = data.get("conf") or []
    for idx, raw_text in enumerate(data.get("text") or []):
        token = str(raw_text or "").strip()
        if not token:
            continue
        conf_raw = confidences[idx] if idx < len(confidences) else "-1"
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        if conf < 25 and not re.search(r"\d", token):
            continue
        tokens.append(token)
        try:
            box_w = max(0, int(data["width"][idx]))
            box_h = max(0, int(data["height"][idx]))
            text_area += box_w * box_h
        except Exception:
            pass

    joined = " ".join(tokens).lower()
    details["token_count"] = len(tokens)
    details["char_count"] = sum(len(re.sub(r"\s+", "", token)) for token in tokens)
    details["text_area_ratio"] = min(1.0, text_area / float(width * height))
    details["keyword_hits"] = len(POSTER_TEXT_RE.findall(joined))
    details["datetime_hits"] = len(DATE_TIME_RE.findall(joined))

    poster_like = False
    if details["text_area_ratio"] >= min_text_area and (
        details["char_count"] >= min_text_chars or details["keyword_hits"] >= 1
    ):
        poster_like = True
    elif details["token_count"] >= min_text_items and (
        details["keyword_hits"] >= 1 or details["datetime_hits"] >= 2
    ):
        poster_like = True
    elif details["keyword_hits"] >= 2 and (
        details["datetime_hits"] >= 1 or details["char_count"] >= min_text_chars
    ):
        poster_like = True

    return poster_like, details


def mean_smallest_distances(values: Sequence[float], count: int) -> float:
    if not values:
        return float("inf")
    limit = max(1, min(int(count), len(values)))
    arr = np.asarray(values, dtype=float)
    smallest = np.partition(arr, limit - 1)[:limit]
    return float(np.mean(smallest))


def _face_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    top = max(a[0], b[0])
    right = min(a[1], b[1])
    bottom = min(a[2], b[2])
    left = max(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(1, (a[1] - a[3]) * (a[2] - a[0]))
    area_b = max(1, (b[1] - b[3]) * (b[2] - b[0]))
    return intersection / float(area_a + area_b - intersection)


def dedupe_face_locations(
    locations: Sequence[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    unique: List[Tuple[int, int, int, int]] = []
    for loc in locations:
        if any(_face_iou(loc, existing) >= 0.35 for existing in unique):
            continue
        unique.append(loc)
    return unique


def detect_faces_tiled(
    pil_img: Image.Image,
    model: str,
    base_upsample: int,
) -> List[Tuple[int, int, int, int]]:
    width, height = pil_img.size
    if max(width, height) < 900:
        return []

    tile_w = min(width, max(640, int(width * 0.72)))
    tile_h = min(height, max(640, int(height * 0.72)))
    x_positions = sorted({0, max(0, width - tile_w)})
    y_positions = sorted({0, max(0, height - tile_h)})

    all_locations: List[Tuple[int, int, int, int]] = []
    for left in x_positions:
        for top in y_positions:
            tile = pil_img.crop((left, top, left + tile_w, top + tile_h))
            arr = np.array(ImageOps.autocontrast(tile).convert("RGB"))
            try:
                tile_locations = face_recognition.face_locations(
                    arr,
                    model=model,
                    number_of_times_to_upsample=max(1, base_upsample),
                )
            except Exception:
                continue
            for t, r, b, l in tile_locations:
                all_locations.append((t + top, r + left, b + top, l + left))

    return dedupe_face_locations(all_locations)


def detect_faces_robust(
    pil_img: Image.Image,
    model: str,
    base_upsample: int,
) -> Tuple[List, np.ndarray, str]:
    """
    複数の前処理を試して顔検出を行う。
    Returns: (locations, numpy_image, method_name)
    """
    # 1. Standard approach with AutoContrast
    img_std = ImageOps.autocontrast(pil_img)
    arr_std = np.array(img_std)
    
    # リサイズ戦略: 小さすぎる画像は大きくする
    orig_w, orig_h = pil_img.size
    min_dim = min(orig_w, orig_h)
    
    # 通常のアップサンプル設定
    upsample = base_upsample
    if min_dim < 400:
        upsample += 1
    
    try:
        locs = face_recognition.face_locations(arr_std, model=model, number_of_times_to_upsample=upsample)
        if locs:
            return locs, arr_std, "std"
    except MemoryError:
        pass

    # 2. Histogram Equalization (for dark/backlit images)
    try:
        img_eq = ImageOps.equalize(pil_img)
        arr_eq = np.array(img_eq)
        locs = face_recognition.face_locations(arr_eq, model=model, number_of_times_to_upsample=upsample)
        if locs:
            logging.debug("Faces found via Histogram Equalization.")
            return locs, arr_eq, "equalize"
    except Exception:
        pass

    # 3. Aggressive Upscale (最後の手段)
    # すでに標準で試しているので、ここではさらに大きくして試す
    if min_dim < 800:
        target_min = 1000
        scale = target_min / float(min_dim)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        try:
            img_big = pil_img.resize((new_w, new_h), Image.LANCZOS)
            # コントラストも強調
            enhancer = ImageEnhance.Contrast(img_big)
            img_big = enhancer.enhance(1.5)
            arr_big = np.array(img_big)
            
            # 画像が大きいのでupsampleは0か1で良い
            locs = face_recognition.face_locations(arr_big, model=model, number_of_times_to_upsample=1)
            if locs:
                logging.debug("Faces found via Aggressive Upscale.")
                return locs, arr_big, "upscale"
        except Exception:
            pass
            
    # 4. CNN Fallback (if not used initially and memory allows)
    if model != "cnn":
        try:
            # CNNはメモリ食うが、精度優先でサイズ制限を緩和 (800 -> 1200)
            img_cnn = pil_img
            w, h = pil_img.size
            max_dim = max(w, h)
            if max_dim > 1200:
                scale = 1200.0 / max_dim
                img_cnn = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            
            arr_cnn = np.array(img_cnn)
            locs = face_recognition.face_locations(arr_cnn, model="cnn", number_of_times_to_upsample=0)
            if locs:
                logging.debug("Faces found via CNN fallback.")
                return locs, arr_cnn, "cnn_fallback"
        except Exception:
            pass

    return [], arr_std, "failed"


def _filter_image_legacy(
    data: bytes,
    known_encodings: Sequence,
    negative_encodings: Sequence,
    tolerance: float,
    negative_tolerance: float,
    negative_margin: float,
    max_faces: int,
    num_jitters: int,
    enforce_two_faces: bool,
    detect_upsample: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    try:
        pil_img = Image.fromarray(face_recognition.load_image_file(io.BytesIO(data))).convert("RGB")
    except Exception:
        return False, "load_error", {}

    # ロバストな顔検出を実行
    locations, image_array, method = detect_faces_robust(pil_img, FACE_MODEL, detect_upsample)

    # -------------------------------------------------------------------------
    # 【追加】ツーショット（複数顔）検出時のCNNスイッチ
    # HOGでちょうど2人検出された場合のみ、精度向上のためCNNで再検出を行う
    # 3人以上はmax_facesで弾かれる前提なのでリソース節約のためスキップ
    # -------------------------------------------------------------------------
    if FACE_MODEL == "hog" and len(locations) == 2:
        logging.debug("HOG found 2 faces. Switching to CNN for better accuracy...")
        try:
            # CNNはメモリ消費が激しいので、画像が大きすぎる場合は縮小する
            h, w = image_array.shape[:2]
            max_dim = max(h, w)
            cnn_input = image_array
            scale_factor = 1.0
            
            if max_dim > 1200: # 精度優先で1200pxまで許容 (元1000)
                scale_factor = 1200.0 / max_dim
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                # PIL経由でリサイズ
                cnn_input = np.array(Image.fromarray(image_array).resize((new_w, new_h), Image.LANCZOS))
                logging.debug("Resized for CNN: %dx%d -> %dx%d", w, h, new_w, new_h)

            cnn_locs = face_recognition.face_locations(cnn_input, model="cnn", number_of_times_to_upsample=0)
            
            if cnn_locs:
                # リサイズしていた場合は座標を元に戻す
                if scale_factor != 1.0:
                    scaled_locs = []
                    for (top, right, bottom, left) in cnn_locs:
                        scaled_locs.append((
                            int(top / scale_factor),
                            int(right / scale_factor),
                            int(bottom / scale_factor),
                            int(left / scale_factor)
                        ))
                    locations = scaled_locs
                else:
                    locations = cnn_locs
                
                method += "+cnn_switch"
                logging.debug("CNN switch successful. Found %d faces.", len(locations))
            else:
                logging.debug("CNN found no faces. Keeping HOG results.")
        except Exception as e:
            logging.debug("CNN switch failed (err=%s). Keeping HOG results.", e)

    # --- 内部関数: 顔の検証ロジック ---
    def validate_faces(current_locs, current_img, encs=None):
        details = {"score": 9.99, "neg_score": 9.99}
        if not current_locs:
            return False, "no_face", False, details # needs_retry
        
        if len(current_locs) > max_faces:
            return False, "too_many_faces", False, details
        
        if enforce_two_faces and len(current_locs) != 2:
            return False, "too_many_faces", False, details

        if encs is None:
            try:
                encs = face_recognition.face_encodings(
                    current_img, current_locs, num_jitters=num_jitters
                )
            except Exception:
                return False, "encode_fail", False, details

        if not encs:
            return False, "encode_fail", False, details

        has_valid_face = False
        retry_candidate = False # 惜しい（本人判定だがNGに近い）場合にTrue
        
        global_best_score = float('inf')
        global_best_neg = float('inf')
        
        # 本人スコアがこれ以下なら、Negative判定を無視して強制採用する閾値
        # 0.35だと緩すぎるため、0.30に下げつつ、Negativeとの乖離チェックを追加
        # 他人が0.281を出して通過してしまったため、さらに厳しく0.27へ
        STRONG_MATCH_THRESHOLD = 0.27
        STRONG_MATCH_NEG_DIFF = 0.04

        for cand in encs:
            dists = face_recognition.face_distance(known_encodings, cand)
            if len(dists) == 0: continue
            score = min(dists)
            global_best_score = min(global_best_score, score)

            neg_score = float('inf')
            if negative_encodings:
                neg_dists = face_recognition.face_distance(negative_encodings, cand)
                if len(neg_dists) > 0:
                    neg_score = min(neg_dists)
                global_best_neg = min(global_best_neg, neg_score)

            # A. 本人似ではない -> スキップ
            if score > tolerance:
                continue

            # 特例: 本人スコアが非常に良い場合は即採用 (Strong Match)
            if score <= STRONG_MATCH_THRESHOLD:
                 # Negativeが本人より大幅に良い値(0.04以上差がある)なら、StrongMatchでも弾く
                 if negative_encodings and neg_score < (score - STRONG_MATCH_NEG_DIFF):
                     logging.debug("Rejected Strong Match: closer to negative (score=%.3f, neg=%.3f)", score, neg_score)
                     retry_candidate = True # 怪しいのでCNNで再検査したい
                 else:
                     logging.debug("Found valid face (Strong Match): score=%.3f", score)
                     has_valid_face = True
                     break
            
            # B. 本人似だが、Negativeチェック
            is_negative = False
            if negative_tolerance > 0 and neg_score <= negative_tolerance:
                # Negative圏内。本人スコアと比較して救済できるか？
                # 「本人スコア < Negativeスコア - マージン」なら本人とみなす
                if score < neg_score - negative_margin:
                    pass 
                else:
                    is_negative = True
            
            if neg_score < score:
                is_negative = True

            if not is_negative:
                logging.debug("Found valid face: score=%.3f, neg=%.3f", score, neg_score)
                has_valid_face = True
                break
            else:
                # 本人圏内に入っているのにNegative判定された -> 惜しいので再検査候補
                retry_candidate = True
        
        details["score"] = global_best_score
        details["neg_score"] = global_best_neg

        if has_valid_face:
            return True, "ok", False, details
        
        if global_best_score > tolerance:
            logging.debug("Rejected: best match %.3f > tolerance %.3f", global_best_score, tolerance)
            return False, "no_match", False, details
        else:
            logging.debug("Rejected: ambiguous or negative match (score=%.3f, neg=%.3f)", global_best_score, global_best_neg)
            return False, "negative_match", retry_candidate, details

    # --- 1回目の検証 (HOG / Initial CNN) ---
    ok, reason, needs_retry, _ = validate_faces(locations, image_array)
    if ok:
        return True, reason, {}

    # --- 2回目の検証 (セカンドオピニオン) ---
    # HOGで「惜しい」判定だった場合、CNNで再挑戦する
    if needs_retry and "cnn" not in method and FACE_MODEL == "hog":
        logging.debug("HOG results ambiguous (potential match rejected). Retrying with CNN...")
        try:
            h, w = image_array.shape[:2]
            max_dim = max(h, w)
            cnn_input = image_array
            scale_factor = 1.0
            
            if max_dim > 1200:
                scale_factor = 1200.0 / max_dim
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                cnn_input = np.array(Image.fromarray(image_array).resize((new_w, new_h), Image.LANCZOS))
            
            cnn_locs = face_recognition.face_locations(cnn_input, model="cnn", number_of_times_to_upsample=0)
            
            if cnn_locs:
                if scale_factor != 1.0:
                    scaled_locs = []
                    for (t, r, b, l) in cnn_locs:
                        scaled_locs.append((
                            int(t/scale_factor), int(r/scale_factor), int(b/scale_factor), int(l/scale_factor)
                        ))
                    locations = scaled_locs
                else:
                    locations = cnn_locs
                
                # 再検証
                ok_retry, reason_retry, _, _ = validate_faces(locations, image_array)
                if ok_retry:
                    logging.debug("CNN retry successful!")
                    return True, "ok", {}
                else:
                    logging.debug("CNN retry result: %s", reason_retry)
                    return False, reason_retry, {}
            else:
                logging.debug("CNN found no faces during retry.")
        except Exception as e:
            logging.debug("CNN retry crashed: %s", e)

    return False, reason, {}


def filter_image(
    data: bytes,
    known_encodings: Sequence,
    negative_encodings: Sequence,
    tolerance: float,
    negative_tolerance: float,
    negative_margin: float,
    max_faces: int,
    num_jitters: int,
    enforce_two_faces: bool,
    detect_upsample: int,
    ocr_enabled: bool,
    ocr_min_text_items: int,
    ocr_min_text_chars: int,
    ocr_min_text_area: float,
    match_profile: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    try:
        pil_img = Image.fromarray(face_recognition.load_image_file(io.BytesIO(data))).convert("RGB")
    except Exception:
        return False, "load_error", {}

    if ocr_enabled:
        poster_like, poster_details = detect_poster_text(
            pil_img,
            ocr_min_text_items,
            ocr_min_text_chars,
            ocr_min_text_area,
        )
        if poster_like:
            logging.debug("Rejected poster-like image via OCR: %s", poster_details)
            return False, "poster_text", poster_details

    locations, image_array, method = detect_faces_robust(pil_img, FACE_MODEL, detect_upsample)

    if FACE_MODEL == "hog" and len(locations) == 2:
        logging.debug("HOG found 2 faces. Switching to CNN for better accuracy...")
        try:
            h, w = image_array.shape[:2]
            max_dim = max(h, w)
            cnn_input = image_array
            scale_factor = 1.0
            if max_dim > 1200:
                scale_factor = 1200.0 / max_dim
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                cnn_input = np.array(Image.fromarray(image_array).resize((new_w, new_h), Image.LANCZOS))
                logging.debug("Resized for CNN: %dx%d -> %dx%d", w, h, new_w, new_h)

            cnn_locs = face_recognition.face_locations(
                cnn_input, model="cnn", number_of_times_to_upsample=0
            )
            if cnn_locs:
                if scale_factor != 1.0:
                    scaled_locs = []
                    for top, right, bottom, left in cnn_locs:
                        scaled_locs.append(
                            (
                                int(top / scale_factor),
                                int(right / scale_factor),
                                int(bottom / scale_factor),
                                int(left / scale_factor),
                            )
                        )
                    locations = scaled_locs
                else:
                    locations = cnn_locs
                method += "+cnn_switch"
                logging.debug("CNN switch successful. Found %d faces.", len(locations))
            else:
                logging.debug("CNN found no faces. Keeping HOG results.")
        except Exception as exc:
            logging.debug("CNN switch failed (err=%s). Keeping HOG results.", exc)

    def validate_faces(current_locs, current_img, encs=None):
        details = {
            "score": 9.99,
            "neg_score": 9.99,
            "support_score": 9.99,
            "face_count": len(current_locs),
            "method": method,
            "profile": match_profile,
        }
        if not current_locs:
            return False, "no_face", False, details

        if len(current_locs) > max_faces:
            return False, "too_many_faces", False, details

        if enforce_two_faces and len(current_locs) != 2:
            return False, "too_many_faces", False, details

        if encs is None:
            try:
                encs = face_recognition.face_encodings(
                    current_img, current_locs, num_jitters=num_jitters
                )
            except Exception:
                return False, "encode_fail", False, details

        if not encs:
            return False, "encode_fail", False, details

        has_valid_face = False
        retry_candidate = False
        global_best_score = float("inf")
        global_best_neg = float("inf")
        global_best_support = float("inf")
        strong_match_threshold = 0.27
        strong_match_neg_diff = 0.04

        for cand in encs:
            dists = face_recognition.face_distance(known_encodings, cand)
            if len(dists) == 0:
                continue
            score = min(dists)
            global_best_score = min(global_best_score, score)
            support_score = mean_smallest_distances(dists, 3)
            global_best_support = min(global_best_support, support_score)

            neg_score = float("inf")
            if negative_encodings:
                neg_dists = face_recognition.face_distance(negative_encodings, cand)
                if len(neg_dists) > 0:
                    neg_score = min(neg_dists)
                global_best_neg = min(global_best_neg, neg_score)

            if score > tolerance:
                continue

            if score <= strong_match_threshold:
                if negative_encodings and neg_score < (score - strong_match_neg_diff):
                    logging.debug(
                        "Rejected Strong Match: closer to negative (score=%.3f, neg=%.3f)",
                        score,
                        neg_score,
                    )
                    retry_candidate = True
                else:
                    logging.debug("Found valid face (Strong Match): score=%.3f", score)
                    has_valid_face = True
                    break

            if match_profile == "noka" and len(dists) >= 3 and score > 0.30:
                support_threshold = min(0.47, tolerance - 0.03)
                if support_score > support_threshold:
                    logging.debug(
                        "Rejected noka candidate: weak support cluster (score=%.3f, support=%.3f, threshold=%.3f)",
                        score,
                        support_score,
                        support_threshold,
                    )
                    retry_candidate = True
                    continue

            is_negative = False
            if negative_tolerance > 0 and neg_score <= negative_tolerance:
                if score >= neg_score - negative_margin:
                    is_negative = True

            if (
                match_profile == "noka"
                and negative_encodings
                and score > strong_match_threshold
                and neg_score < float("inf")
            ):
                min_gap = max(negative_margin, 0.035)
                if neg_score <= score + min_gap:
                    is_negative = True

            if neg_score < score:
                is_negative = True

            if not is_negative:
                logging.debug("Found valid face: score=%.3f, neg=%.3f", score, neg_score)
                has_valid_face = True
                break

            retry_candidate = True

        details["score"] = global_best_score
        details["neg_score"] = global_best_neg
        details["support_score"] = global_best_support

        if has_valid_face:
            return True, "ok", False, details

        if global_best_score > tolerance:
            logging.debug(
                "Rejected: best match %.3f > tolerance %.3f",
                global_best_score,
                tolerance,
            )
            return False, "no_match", False, details

        logging.debug(
            "Rejected: ambiguous or negative match (score=%.3f, neg=%.3f)",
            global_best_score,
            global_best_neg,
        )
        return False, "negative_match", retry_candidate, details

    ok, reason, needs_retry, _ = validate_faces(locations, image_array)
    if ok:
        return True, reason, {}

    if needs_retry and "cnn" not in method and FACE_MODEL == "hog":
        logging.debug("HOG results ambiguous (potential match rejected). Retrying with CNN...")
        try:
            h, w = image_array.shape[:2]
            max_dim = max(h, w)
            cnn_input = image_array
            scale_factor = 1.0
            if max_dim > 1200:
                scale_factor = 1200.0 / max_dim
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                cnn_input = np.array(Image.fromarray(image_array).resize((new_w, new_h), Image.LANCZOS))

            cnn_locs = face_recognition.face_locations(
                cnn_input, model="cnn", number_of_times_to_upsample=0
            )
            if cnn_locs:
                if scale_factor != 1.0:
                    scaled_locs = []
                    for top, right, bottom, left in cnn_locs:
                        scaled_locs.append(
                            (
                                int(top / scale_factor),
                                int(right / scale_factor),
                                int(bottom / scale_factor),
                                int(left / scale_factor),
                            )
                        )
                    locations = scaled_locs
                else:
                    locations = cnn_locs

                ok_retry, reason_retry, _, _ = validate_faces(locations, image_array)
                if ok_retry:
                    logging.debug("CNN retry successful!")
                    return True, "ok", {}

                logging.debug("CNN retry result: %s", reason_retry)
                return False, reason_retry, {}

            logging.debug("CNN found no faces during retry.")
        except Exception as exc:
            logging.debug("CNN retry crashed: %s", exc)

    return False, reason, {}


def download_image(url: str, timeout: float, user_agent: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.debug)
    global FACE_UPSAMPLE, FACE_MODEL
    FACE_UPSAMPLE = max(0, int(args.upsample))
    FACE_MODEL = args.model

    out_dir = Path(args.out_dir)
    ref_dir = Path(args.reference_dir)
    neg_dir = Path(args.negative_dir) if args.negative_dir else None
    usage_file = Path(args.usage_file)
    usage_state = load_usage_state(usage_file)
    usage_changed = False

    if args.ocr_enabled and not pytesseract:
        logging.warning("OCR requested but pytesseract is unavailable; poster filtering is disabled.")

    ensure_out_dir(out_dir)
    deduplicate_images(out_dir)
    if ref_dir != out_dir:
        deduplicate_images(ref_dir)
    if neg_dir and neg_dir.exists() and neg_dir != out_dir:
        deduplicate_images(neg_dir)

    if args.clean_training_dirs:
        _clean_face_dir(ref_dir, args.num_jitters, "reference")
        if neg_dir:
            _clean_face_dir(neg_dir, args.num_jitters, "negative")

    load_known_encodings._workers = args.workers
    load_negative_encodings._workers = args.workers
    _clean_face_dir._workers = args.workers

    known_encodings = load_known_encodings(ref_dir, args.num_jitters, args.cache_only)
    negative_encodings = (
        load_negative_encodings(neg_dir, args.num_jitters, args.cache_only)
        if neg_dir
        else []
    )
    existing_hashes = digests_for_existing(out_dir)
    seen_hashes: Set[str] = set(existing_hashes)
    blocked_digests = load_block_list(Path(args.block_list)) if args.block_list else set()

    saved = 0
    new_saved_files: List[Path] = []
    seen_urls: Set[str] = set()
    stats = defaultdict(int)

    max_age_cutoff: Optional[datetime] = None
    if args.max_age_days > 0:
        max_age_cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)

    def handle_image_items(
        image_items: List[Tuple[str, Optional[int]]],
        label: str,
        tolerance: float,
        neg_tol: float,
        neg_margin: float,
        max_faces: int,
        enforce_two_faces: bool,
        match_profile: str,
    ) -> bool:
        nonlocal saved, usage_changed
        # 検出用アップサンプル数の自動調整
        detect_upsample = FACE_UPSAMPLE
        if "のか" in label:
            detect_upsample = max(detect_upsample, 1) # 最低でも1回はアップサンプル

        new_items: List[Tuple[str, Optional[int]]] = []
        page_seen: Set[str] = set()
        hit_old = 0
        for url, created_at in image_items:
            if url in seen_urls or url in page_seen:
                continue
            page_seen.add(url)
            seen_urls.add(url)
            if max_age_cutoff:
                if created_at is None:
                    # 日付不明でも一旦許可（厳しすぎると取りこぼすため）
                    pass
                else:
                    dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
                    if dt < max_age_cutoff:
                        hit_old += 1
                        if hit_old <= 3:
                            logging.debug("Skip old image (%s): %s", dt.date(), url)
                        continue
            new_items.append((url, created_at))
        
        if not new_items:
            logging.debug("No unseen images for %s; stopping.", label)
            return False

        total = len(new_items)
        for idx, (url, _) in enumerate(new_items, 1):
            logging.debug("Processing %s image %d/%d: %s", label, idx, total, url)
            try:
                data = download_image(url, args.timeout, args.user_agent)
            except Exception:
                logging.warning("Failed to download %s; skipping.", url)
                stats["download_fail"] += 1
                continue

            digest = compute_digest(data)
            if blocked_digests and any(digest.startswith(b) for b in blocked_digests):
                logging.debug("Skip blocked digest prefix: %s (url=%s)", digest[:12], url)
                stats["blocked"] += 1
                continue
            if digest in seen_hashes:
                logging.debug("Skip duplicate (hash match): %s", url)
                stats["duplicate_hash"] += 1
                continue
            seen_hashes.add(digest)

            ext = determine_extension(data)
            if not ext or f".{ext.lstrip('.')}" not in IMAGE_EXTS:
                logging.debug("Rejected: unsupported image type from %s", url)
                stats["unsupported"] += 1
                continue

            ok, reason, _ = filter_image(
                data,
                known_encodings,
                negative_encodings,
                tolerance,
                neg_tol,
                neg_margin,
                max_faces,
                args.num_jitters,
                enforce_two_faces,
                detect_upsample,
                args.ocr_enabled,
                args.ocr_min_text_items,
                args.ocr_min_text_chars,
                args.ocr_min_text_area,
                match_profile,
            )
            if not ok:
                stats[reason] += 1
                continue

            filename = f"{FILENAME_PREFIX}_{digest[:12]}{ext}"
            save_path = out_dir / filename
            with open(save_path, "wb") as f:
                f.write(data)
            existing_hashes.add(digest)
            saved += 1
            stats["saved"] += 1
            logging.info("判定OK [%s] -> %s", label, save_path)
            if ensure_usage_entry(usage_state, filename, time.time()):
                usage_changed = True
            new_saved_files.append(save_path)
        
        if hit_old >= 5: # 古い画像が5枚続いたら停止（バッファを持たせる）
            logging.debug(
                "Encountered multiple old images for %s; stopping further pages.", label
            )
            return False
        return True

    if args.html_file:
        try:
            html = Path(args.html_file).read_text(encoding="utf-8")
        except Exception:
            logging.exception("Failed to read html file: %s", args.html_file)
            return
        base_url = args.base_url or ""
        image_items, _ = extract_image_urls(html, base_url)
        handle_image_items(
            image_items,
            "html",
            args.tolerance,
            args.negative_tolerance,
            args.negative_margin,
            args.max_faces,
            False,
            "default",
        )
        if usage_changed:
            save_usage_state(usage_file, usage_state)
        logging.info("Finished. Saved %d new images.", saved)
        return

    for source_url in args.urls:
        # NOTE: 以前のハードコードされた値を緩和し、CLI引数を尊重するように変更
        # 本人の基準値（少し甘め）
        base_tolerance = args.tolerance
        base_neg_tol = args.negative_tolerance
        match_profile = "default"
        
        if "MEGAFON_noka" in source_url:
            match_profile = "noka"
            source_label = "のか"
            tol = base_tolerance + 0.05  # 本人はさらに甘く (例: 0.50 -> 0.55)
            neg_tol = 0.0 # NG判定なし
            neg_margin = 0.03 # ローカル実験結果反映: 0.0 -> 0.03
            max_faces = 2  # ツーショまで
            enforce_two_faces = False
        elif "MEGAFON_idol" in source_url:
            source_label = "のか(公式)"
            # 公式は少し厳しめ＋ネガティブも併用して誤検知を減らす
            tol = base_tolerance + 0.02  # 本人より少し甘い程度
            neg_tol = 0.23              # ネガティブ基準を有効にする
            neg_margin = 0.02           # ベストとの差が小さいときは弾く
            max_faces = 2  # ツーショまで
            enforce_two_faces = False
        else:
            source_label = "その他(ツーショ限定)"
            tol = base_tolerance # 標準
            neg_tol = base_neg_tol
            # ツーショット（その他）の場合は、救済措置を効かせるためマージンを少し小さめに
            neg_margin = 0.02 
            max_faces = 2
            enforce_two_faces = True # 修正: 2人写っている場合のみ許可する

        parsed = urlparse(source_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        cursor: Optional[str] = None
        logging.info("Scraping %s (tol=%.2f)", source_url, tol)

        while True:
            try:
                html = fetch_search_page(
                    source_url, cursor, args.timeout, args.user_agent
                )
            except Exception:
                logging.exception("Failed to fetch page: %s", source_url)
                break

            image_items, next_cursor = extract_image_urls(html, base_url)
            if not handle_image_items(
                image_items,
                source_label,
                tol,
                neg_tol,
                neg_margin,
                max_faces,
                enforce_two_faces,
                match_profile,
            ):
                break

            if not next_cursor:
                logging.info("No next cursor for %s; moving on.", source_url)
                break
            if cursor == next_cursor:
                logging.info("Cursor did not advance for %s; stopping to avoid loop.", source_url)
                break
            cursor = next_cursor

    logging.info("Finished. Saved %d new images.", saved)
    removed = remove_near_duplicates(out_dir, new_saved_files)
    stats["near_dup_removed"] += removed
    for path in new_saved_files:
        if path.exists():
            continue
        if path.name in usage_state:
            usage_state.pop(path.name, None)
            usage_changed = True
    if usage_changed:
        save_usage_state(usage_file, usage_state)
    logging.info(
        "Summary: saved=%d, near_dup_removed=%d, blocked=%d, duplicate_hash=%d, unsupported=%d, download_fail=%d, poster_text=%d, no_face=%d, too_many_faces=%d, encode_fail=%d, no_match=%d",
        stats["saved"],
        stats["near_dup_removed"],
        stats["blocked"],
        stats["duplicate_hash"],
        stats["unsupported"],
        stats["download_fail"],
        stats["poster_text"],
        stats["no_face"],
        stats["too_many_faces"],
        stats["encode_fail"],
        stats["no_match"],
    )


if __name__ == "__main__":
    main()
