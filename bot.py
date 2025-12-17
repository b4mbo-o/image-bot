#!/usr/bin/env python3
import argparse
import fcntl
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import tweepy


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DEFAULT_HISTORY_SIZE = 12
DEFAULT_ENV_FILE = ".env"
DEFAULT_USAGE_FILE = "state/usage.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post an image to Twitter, avoiding the last N images."
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Directory that holds candidate images.",
    )
    parser.add_argument(
        "--history-file",
        default="state/history.json",
        help="Path to store recent post history.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=DEFAULT_HISTORY_SIZE,
        help="How many recent images to keep out of rotation.",
    )
    parser.add_argument(
        "--usage-file",
        default=DEFAULT_USAGE_FILE,
        help="Path to store per-image usage counts.",
    )
    parser.add_argument(
        "--prefer-rare",
        action="store_true",
        default=True,
        help="Prefer rarely used images when selecting (default: on).",
    )
    parser.add_argument(
        "--no-prefer-rare",
        dest="prefer_rare",
        action="store_false",
        help="Disable preference for rarely used images.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=4.0,
        help="Interval (hours) between posts when --loop is set.",
    )
    parser.add_argument(
        "--text",
        default="",
        help="Tweet text. Leave empty to post only the image.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running and post every --interval-hours.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except the actual tweet or history update.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="Path to .env file with Twitter credentials.",
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


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    loaded = 0
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        logging.info("Loaded %d entries from %s", loaded, env_path)


def build_twitter_clients():
    try:
        consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
        consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]
        access_token = os.environ["TWITTER_ACCESS_TOKEN"]
        access_token_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
    except KeyError as exc:
        raise RuntimeError(
            f"Missing environment variable: {exc.args[0]}"
        ) from exc

    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret, access_token, access_token_secret
    )
    api = tweepy.API(auth)
    client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=True,
    )
    return client, api


def load_history(path: Path) -> List[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "recent" in data:
            data = data["recent"]
        if not isinstance(data, list):
            raise ValueError("history file is malformed")
        return [str(item) for item in data]
    except json.JSONDecodeError:
        logging.warning("History file is empty or invalid JSON; resetting: %s", path)
        return []


def load_usage(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        raise ValueError("usage file is malformed")
    except json.JSONDecodeError:
        logging.warning("Usage file is empty or invalid JSON; resetting: %s", path)
        return {}


def save_usage(path: Path, usage: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def save_history(path: Path, history: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"recent": list(history)}, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def select_image(
    images: Sequence[Path],
    recent: Sequence[str],
    history_size: int,
    usage: Dict[str, Dict[str, float]],
    prefer_rare: bool,
) -> Path:
    if not images:
        raise RuntimeError("No images found to post.")

    history_tail = list(recent)[-history_size:]
    recent_set = set(history_tail)
    eligible = [p for p in images if p.name not in recent_set]

    if not eligible:
        logging.info(
            "All images are in the recent list; allowing reuse for this post."
        )
        eligible = list(images)

    pool = eligible
    if prefer_rare and usage:
        min_count = min(usage.get(p.name, {}).get("count", 0) for p in eligible)
        rare = [p for p in eligible if usage.get(p.name, {}).get("count", 0) == min_count]
        if rare:
            pool = rare
            logging.info("Prefer rare images: choosing among %d least-used.", len(pool))

    chosen = random.choice(pool)
    return chosen


def upload_and_tweet(
    client: tweepy.Client, api: tweepy.API, image_path: Path, text: str
) -> str:
    media = api.media_upload(filename=str(image_path))
    result = client.create_tweet(text=text, media_ids=[media.media_id])
    tweet_id = str(result.data.get("id", ""))
    return tweet_id


@contextmanager
def file_lock(lock_path: Path) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def run_once(args: argparse.Namespace) -> None:
    load_env(Path(args.env_file))
    images_dir = Path(args.images_dir)
    history_file = Path(args.history_file)
    usage_file = Path(args.usage_file)
    lock_file = history_file.with_suffix(history_file.suffix + ".lock")

    with file_lock(lock_file):
        history = load_history(history_file)
        usage = load_usage(usage_file)
        images = list_images(images_dir)
        logging.info("Images available in %s: %d", images_dir, len(images))
        chosen = select_image(images, history, args.history_size, usage, args.prefer_rare)
        rel_name = chosen.relative_to(images_dir).as_posix()

        logging.info("Selected image: %s", rel_name)

        if args.dry_run:
            logging.info("Dry-run mode: skipping tweet and history update.")
            return

        client, api = build_twitter_clients()
        tweet_id = upload_and_tweet(client, api, chosen, args.text)
        logging.info("Tweet posted: id=%s image=%s", tweet_id or "<unknown>", rel_name)

        history.append(rel_name)
        trimmed = history[-args.history_size :]
        save_history(history_file, trimmed)
        logging.info("History updated; %d entries retained.", len(trimmed))

        now_ts = time.time()
        usage_entry = usage.get(rel_name) or {"count": 0, "last_used": 0}
        usage_entry["count"] = usage_entry.get("count", 0) + 1
        usage_entry["last_used"] = now_ts
        usage[rel_name] = usage_entry
        save_usage(usage_file, usage)
        logging.info("Usage updated; count=%d for %s", usage_entry["count"], rel_name)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    if args.loop:
        logging.info(
            "Starting loop: posting every %.2f hours. Press Ctrl+C to stop.",
            args.interval_hours,
        )
        while True:
            try:
                run_once(args)
            except KeyboardInterrupt:
                logging.info("Interrupted; exiting.")
                sys.exit(0)
            except Exception:
                logging.exception("Error during run; continuing after sleep.")
            time.sleep(args.interval_hours * 3600)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
