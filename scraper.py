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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import face_recognition
import requests
from bs4 import BeautifulSoup
from PIL import Image
from PIL import ImageChops
from zoneinfo import ZoneInfo
from collections import defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
FILENAME_PREFIX = "yahoo_rt"
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
        help="Yahoo!リアルタイム検索などのページURL（例: https://search.yahoo.co.jp/...）。",
    )
    parser.add_argument(
        "--html-file",
        default="",
        help="保存済みHTMLファイルを解析してダウンロードする場合のパス（ネット取得なし）。",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="--html-file使用時の基点URL（相対リンク解決用、不要なら空でOK）。",
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
        help="除外したい顔画像ディレクトリ（NG顔、空なら無効）。",
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
        help="学習キャッシュ(.encodings_cache.pkl)を優先して使用する（デフォルトON）。",
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
        default=3,
        help="実行日からさかのぼって何日までの投稿を対象にするか（0以下で無効）。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.45,
        help="顔マッチの厳しさ（小さいほど厳密。face_recognition標準は0.6）。",
    )
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=0.35,
        help="NG顔とみなす距離（これ以下だと弾く、0以下で無効）。",
    )
    parser.add_argument(
        "--negative-margin",
        type=float,
        default=0.03,
        help="NG顔と紛らわしい場合のマージン（best_neg が best_pos + margin より小さい/近いなら弾く）。",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=3,
        help="顔エンコード時のジッター回数（増やすと精度↑・速度↓）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="顔処理に使うプロセス数（未指定なら2、0以下でCPU数）。",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=2,
        help="1枚あたりの許容顔数（多人数写真は除外）。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTPタイムアウト秒数。",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="ログファイルパス（指定しないと標準出力のみ）。",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTPリクエストのUser-Agent。",
    )
    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
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

    # 完全一致のハッシュで重複排除
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
    """
    Remove newly saved files that are perceptually identical
    to an existing image (very small hamming distance).
    """
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
    # "3分前", "2時間前"
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
    # "昨日 22:35" or "今日 08:12"
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
    # "MM月DD日(...) HH:MM" or "MM月DD日 HH:MM"
    m = re.match(r"(\d{1,2})月(\d{1,2})日.*?(\d{1,2}):(\d{2})", text)
    if m:
        month, day, hour, minute = map(int, m.groups())
        year = now.year
        dt = datetime(year, month, day, hour, minute, tzinfo=now.tzinfo)
        if dt > now + timedelta(days=1):
            dt = dt.replace(year=year - 1)
        return int(dt.timestamp())
    # "YYYY年MM月DD日 HH:MM"
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
    # Parallelize encoding build (up to 3 workers).
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
    """
    Remove files that cannot produce any face encodings.
    Uses HOG model for speed.
    """
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
    """
    Worker: returns (path, encodings_list).
    Uses HOG model for CPU speed.
    """
    path = Path(path_str)
    data = path.read_bytes()
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations, num_jitters=num_jitters)
    return path_str, encodings


def _encode_faces_from_path_with_count(path_str: str, num_jitters: int):
    """
    Worker: returns (path, encodings_list, face_count).
    """
    path = Path(path_str)
    data = path.read_bytes()
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations, num_jitters=num_jitters)
    return path_str, encodings, len(locations)


def _has_any_face_in_path(path_str: str) -> bool:
    path = Path(path_str)
    data = path.read_bytes()
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(image, model="hog")
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
                # Yahoo!リアルタイム検索にはcursorが無いのでここで返す
                return urls, None
        except Exception:
            logging.warning(
                "Failed to parse Yahoo! realtime search payload; falling back to HTML.",
                exc_info=True,
            )

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
        if src and "/pic/" in src:
            ts = time_values[time_idx] if time_idx < len(time_values) else None
            time_idx += 1
            urls.append((urljoin(base_url, src), ts))

    next_cursor = None
    more_link = soup.find("a", href=lambda h: h and "cursor=" in h)
    if more_link:
        href = more_link.get("href", "")
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "cursor" in qs and qs["cursor"]:
            next_cursor = qs["cursor"][0]

    return urls, next_cursor


def determine_extension(data: bytes) -> Optional[str]:
    kind = imghdr.what(None, h=data)
    if kind == "jpeg":
        return ".jpg"
    if kind in ("png", "gif", "webp"):
        return f".{kind}"
    return None


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
) -> Tuple[bool, str]:
    image = face_recognition.load_image_file(io.BytesIO(data))
    locations = face_recognition.face_locations(image)
    if not locations:
        logging.info("Rejected: no face detected.")
        return False, "no_face"
    if len(locations) > max_faces:
        logging.info("Rejected: %d faces detected (max %d).", len(locations), max_faces)
        return False, "too_many_faces"
    if enforce_two_faces and len(locations) != 2:
        logging.info("Rejected: require exact two faces, found %d.", len(locations))
        return False, "too_many_faces"
    encodings = face_recognition.face_encodings(
        image, locations, num_jitters=num_jitters
    )
    if not encodings:
        logging.info("Rejected: could not encode faces.")
        return False, "encode_fail"
    min_dists = []
    min_neg_dists = []
    for cand in encodings:
        dists = face_recognition.face_distance(known_encodings, cand)
        if len(dists) == 0:
            continue
        min_dists.append(min(dists))
        if negative_encodings and negative_tolerance > 0:
            neg_dists = face_recognition.face_distance(negative_encodings, cand)
            if len(neg_dists) > 0:
                min_neg_dists.append(min(neg_dists))

    if not min_dists:
        logging.info("Rejected: could not compute face distance.")
        return False, "encode_fail"

    best = min(min_dists)
    if best > tolerance:
        logging.info("Rejected: faces do not match known references (best=%.3f).", best)
        return False, "no_match"

    if min_neg_dists:
        best_neg = min(min_neg_dists)
        # Hard reject if it's too close to a negative identity.
        if negative_tolerance > 0 and best_neg <= negative_tolerance:
            logging.info(
                "Rejected: matches negative references (best_neg=%.3f <= %.3f).",
                best_neg,
                negative_tolerance,
            )
            return False, "negative_match"
        # Also reject if the negative match is closer than the positive match by a margin.
        # This helps when multiple similar faces exist in the group.
        if negative_margin > 0 and best_neg <= best + negative_margin:
            logging.info(
                "Rejected: ambiguous vs negative (best=%.3f, best_neg=%.3f, margin=%.3f).",
                best,
                best_neg,
                negative_margin,
            )
            return False, "negative_ambiguous"
    return True, "ok"


def download_image(url: str, timeout: float, user_agent: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    out_dir = Path(args.out_dir)
    ref_dir = Path(args.reference_dir)
    neg_dir = Path(args.negative_dir) if args.negative_dir else None

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
        enforce_two_faces: bool,
        neg_tol: float,
        neg_margin: float,
    ) -> bool:
        nonlocal saved
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
                    logging.info("Skip image with unknown date (strict mode): %s", url)
                    continue
                dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
                if dt < max_age_cutoff:
                    hit_old += 1
                    if hit_old <= 3:
                        logging.info("Skip old image (%s): %s", dt.date(), url)
                    continue
            new_items.append((url, created_at))
        if not new_items:
            logging.info("No unseen images for %s; stopping.", label)
            return False

        for url, _ in new_items:
            try:
                data = download_image(url, args.timeout, args.user_agent)
            except Exception:
                logging.warning("Failed to download %s; skipping.", url)
                stats["download_fail"] += 1
                continue

            digest = compute_digest(data)
            if digest in seen_hashes:
                logging.info("Skip duplicate (hash match): %s", url)
                stats["duplicate_hash"] += 1
                continue
            seen_hashes.add(digest)

            ext = determine_extension(data)
            if not ext or f".{ext.lstrip('.')}" not in IMAGE_EXTS:
                logging.info("Rejected: unsupported image type from %s", url)
                stats["unsupported"] += 1
                continue

            ok, reason = filter_image(
                data,
                known_encodings,
                negative_encodings,
                tolerance,
                neg_tol,
                neg_margin,
                args.max_faces,
                args.num_jitters,
                enforce_two_faces,
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
            logging.info("Saved %s (total saved: %d)", save_path, saved)
            new_saved_files.append(save_path)
        if hit_old >= 3:
            logging.info("Encountered multiple old images for %s; stopping further pages.", label)
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
        # HTML取得時はURL名でモードを決められないので厳しめ（tolerance same, enforce_two_faces=False）
        handle_image_items(
            image_items,
            args.html_file,
            args.tolerance,
            args.negative_tolerance,
            args.negative_margin,
            False,
        )
        logging.info("Finished. Saved %d new images.", saved)
        return

    for source_url in args.urls:
        is_noka = "MEGAFON_noka" in source_url
        is_idol = "MEGAFON_idol" in source_url

        if is_noka:
            tol = 0.50  # 本人優先で緩め
            neg_tol = 0.40
            neg_margin = 0.03
            enforce_two_faces = False
        elif is_idol:
            tol = 0.45
            neg_tol = 0.32  # ネガティブに近い場合は厳しめ
            neg_margin = 0.03
            enforce_two_faces = False
        else:
            tol = 0.38  # 他メンバーは厳しめ
            neg_tol = 0.32
            neg_margin = 0.02
            enforce_two_faces = True  # のかとの2ショットのみ許可

        parsed = urlparse(source_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        cursor: Optional[str] = None
        logging.info("Scraping %s", source_url)

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
                image_items, source_url, tol, neg_tol, neg_margin, enforce_two_faces
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
    logging.info(
        "Summary: saved=%d, near_dup_removed=%d, duplicate_hash=%d, unsupported=%d, download_fail=%d, no_face=%d, too_many_faces=%d, encode_fail=%d, no_match=%d",
        stats["saved"],
        stats["near_dup_removed"],
        stats["duplicate_hash"],
        stats["unsupported"],
        stats["download_fail"],
        stats["no_face"],
        stats["too_many_faces"],
        stats["encode_fail"],
        stats["no_match"],
    )


if __name__ == "__main__":
    main()
