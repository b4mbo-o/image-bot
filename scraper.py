#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set

import tweepy


DEFAULT_ENV_FILE = ".env"
DEFAULT_COMMANDS = ("削除", "delete", "remove", "消して")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete bot tweets and local images when an authorized reply contains a delete command."
    )
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--history-file", default="state/history.json")
    parser.add_argument("--usage-file", default="state/usage.json")
    parser.add_argument("--posts-file", default="state/posts.json")
    parser.add_argument("--moderation-file", default="state/moderation.json")
    parser.add_argument("--block-list", default="state/block.json")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--delete-command-users",
        default="",
        help="Comma-separated usernames allowed to issue delete commands.",
    )
    parser.add_argument(
        "--delete-commands",
        default=",".join(DEFAULT_COMMANDS),
        help="Comma-separated delete command keywords.",
    )
    parser.add_argument("--log-file", default="")
    parser.add_argument("--dry-run", action="store_true")
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
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def build_twitter_client() -> tweepy.Client:
    try:
        consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
        consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]
        access_token = os.environ["TWITTER_ACCESS_TOKEN"]
        access_token_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
    except KeyError as exc:
        raise RuntimeError(f"Missing environment variable: {exc.args[0]}") from exc

    return tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=True,
    )


def load_json_dict(path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Invalid JSON; resetting: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_history(path: Path) -> List[str]:
    data = load_json_dict(path)
    recent = data.get("recent", data if isinstance(data, list) else [])
    if isinstance(recent, list):
        return [str(item) for item in recent]
    return []


def save_history(path: Path, history: Sequence[str]) -> None:
    save_json(path, {"recent": list(history)})


def load_block_entries(path: Path) -> List[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Invalid block list JSON; rebuilding from empty state: %s", path)
        return []
    if isinstance(raw, dict):
        for key in ("block", "items", "files", "digests"):
            if isinstance(raw.get(key), list):
                return [str(item) for item in raw[key] if item]
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def save_block_entries(path: Path, entries: Sequence[str]) -> None:
    seen: Set[str] = set()
    ordered: List[str] = []
    for entry in entries:
        value = str(entry).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    save_json(path, {"block": ordered})


def parse_csv_set(raw: str) -> Set[str]:
    values = set()
    for item in raw.split(","):
        value = item.strip().lstrip("@").lower()
        if value:
            values.add(value)
    return values


def normalize_post_record(raw: object) -> Dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {"image": raw}
    return {}


def extract_replied_to_tweet_id(tweet) -> str:
    for ref in getattr(tweet, "referenced_tweets", None) or []:
        if getattr(ref, "type", "") == "replied_to":
            return str(ref.id)
    return ""


def is_delete_command(text: str, commands: Sequence[str]) -> bool:
    normalized = re.sub(r"@[A-Za-z0-9_]+", " ", text or "").lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(command and command in normalized for command in commands)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    load_env(Path(args.env_file))

    try:
        client = build_twitter_client()
    except RuntimeError as exc:
        logging.error("Twitter credentials missing or invalid: %s", exc)
        return

    me_response = client.get_me(user_fields=["username"], user_auth=True)
    if not me_response or not me_response.data:
        logging.error("Failed to fetch authenticated user information.")
        return

    me = me_response.data
    my_user_id = str(me.id)
    allowed_users = {str(getattr(me, "username", "")).lower()}
    allowed_users.update(parse_csv_set(os.environ.get("DELETE_COMMAND_USERS", "")))
    allowed_users.update(parse_csv_set(args.delete_command_users))
    delete_commands = [item.strip().lower() for item in args.delete_commands.split(",") if item.strip()]

    images_dir = Path(args.images_dir)
    history_file = Path(args.history_file)
    usage_file = Path(args.usage_file)
    posts_file = Path(args.posts_file)
    moderation_file = Path(args.moderation_file)
    block_list_file = Path(args.block_list)

    history = load_history(history_file)
    usage = load_json_dict(usage_file)
    posts = load_json_dict(posts_file)
    moderation_state = load_json_dict(moderation_file)
    block_entries = load_block_entries(block_list_file)

    since_id = moderation_state.get("last_mention_id")
    response = client.get_users_mentions(
        my_user_id,
        since_id=since_id,
        max_results=100,
        expansions=["author_id"],
        tweet_fields=["author_id", "created_at", "in_reply_to_user_id", "referenced_tweets"],
        user_fields=["username"],
        user_auth=True,
    )

    mentions = list(getattr(response, "data", None) or [])
    includes = getattr(response, "includes", None) or {}
    user_lookup = {
        str(user.id): str(getattr(user, "username", "")).lower()
        for user in includes.get("users", []) or []
    }

    if not mentions:
        logging.info("No new mentions to moderate.")
        return

    history_changed = False
    usage_changed = False
    posts_changed = False
    block_changed = False
    max_seen_id = max(int(str(tweet.id)) for tweet in mentions)

    for mention in sorted(mentions, key=lambda item: int(str(item.id))):
        author_username = user_lookup.get(str(getattr(mention, "author_id", "")), "")
        if author_username not in allowed_users:
            continue
        if not is_delete_command(getattr(mention, "text", ""), delete_commands):
            continue

        replied_to_tweet_id = extract_replied_to_tweet_id(mention)
        if not replied_to_tweet_id:
            continue

        post_record = normalize_post_record(posts.get(replied_to_tweet_id))
        image_name = str(post_record.get("image", "")).strip()
        if not image_name:
            logging.info(
                "Authorized delete command matched tweet %s, but no local post mapping was found.",
                replied_to_tweet_id,
            )
            continue

        image_basename = Path(image_name).name
        image_path = images_dir / image_basename
        logging.info(
            "Delete command accepted from @%s for tweet=%s image=%s",
            author_username,
            replied_to_tweet_id,
            image_basename,
        )

        if args.dry_run:
            continue

        try:
            client.delete_tweet(replied_to_tweet_id, user_auth=True)
        except tweepy.errors.NotFound:
            logging.info("Tweet already deleted on X: %s", replied_to_tweet_id)
        except Exception as exc:
            logging.warning("Failed to delete tweet %s: %s", replied_to_tweet_id, exc)
            continue

        if image_path.exists():
            try:
                image_path.unlink()
                logging.info("Deleted local image: %s", image_path)
            except Exception as exc:
                logging.warning("Failed to delete local image %s: %s", image_path, exc)

        new_history = [
            item
            for item in history
            if Path(str(item)).name != image_basename and str(item) != image_name
        ]
        if new_history != history:
            history = new_history
            history_changed = True

        if usage.pop(image_name, None) is not None:
            usage_changed = True
        if image_basename != image_name and usage.pop(image_basename, None) is not None:
            usage_changed = True

        if image_basename not in block_entries:
            block_entries.append(image_basename)
            block_changed = True

        if posts.pop(replied_to_tweet_id, None) is not None:
            posts_changed = True

    moderation_state["last_mention_id"] = str(max_seen_id)
    save_json(moderation_file, moderation_state)

    if args.dry_run:
        logging.info("Dry-run mode: moderation state only was advanced.")
        return

    if history_changed:
        save_history(history_file, history)
    if usage_changed:
        save_json(usage_file, usage)
    if posts_changed:
        save_json(posts_file, posts)
    if block_changed:
        save_block_entries(block_list_file, block_entries)


if __name__ == "__main__":
    main()
