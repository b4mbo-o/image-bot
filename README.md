# 🤖 Image Bot & Scraper

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

画像を自動投稿する **Twitter (X) Bot** と、Yahoo!リアルタイム検索から特定の画像を収集・選別する **スクレイパー** のオールインワンツールセットです。

直近の投稿履歴を回避する重複防止機能や、顔認識によるスマートな画像フィルタリング機能を搭載しています。

---

## 📖 目次

- [✨ 特徴](#-特徴)
- [📂 ディレクトリ構成](#-ディレクトリ構成)
- [🚀 セットアップ](#-セットアップ)
- [🔖 ブランチと配布物](#-ブランチと配布物)
- [🤖 使い方: 画像投稿 BOT](#-使い方-画像投稿-bot)
- [📷 使い方: 画像スクレイパー](#-使い方-画像スクレイパー)
- [☁️ GitHub Actions 運用](#-github-actions-運用)
- [⏰ ローカル/サーバーでの自動実行](#-ローカルサーバーでの自動実行)
- [⚠️ 注意事項](#-注意事項)

---

## ✨ 特徴

### 🐦 画像投稿 BOT
- **スマートな投稿**: 指定ディレクトリから画像をランダムに選択し投稿。直近の履歴（デフォルト12件）と重複する画像は避けます。
- **柔軟なスケジュール**: 単発実行はもちろん、ループ実行やCronでの定期実行に対応。
- **ドライラン機能**: 実際に投稿せずに動作確認が可能。

### 🔍 画像スクレイパー
- **顔認識フィルタ**: `face_recognition` を使用し、基準となる顔画像（`MEGAFON_noka/`等）と一致する写真のみを保存します。
- **ロバスト検出**: 明るさ補正/リサイズ/アップサンプルを複数試し、ツーショット時はCNN再判定も行うことで検出漏れを減らします。
- **集合写真の除外**: 3人以上の集合写真は自動的にスキップし、ピンショットやツーショットのみを厳選。
- **高度な重複排除**: 完全一致だけでなく、p-hashを用いた近似画像の重複も排除します。

---

## 📂 ディレクトリ構成

```text
.
├── bot.py             # 画像投稿BOT 本体
├── scraper.py         # 画像スクレイパー 本体
├── requirements.txt   # 依存ライブラリ一覧
├── .env               # 環境変数設定ファイル（要作成）
├── images/            # 投稿用画像ディレクトリ（.jpg, .png, .webp等）
├── MEGAFON_noka/      # スクレイピング時の基準顔画像ディレクトリ
├── logs/              # ログ出力先
└── state/
    └── history.json   # 投稿履歴データ
```

---

## 🚀 セットアップ

### 1. リポジトリのクローンと仮想環境の作成

```bash
# リポジトリをクローン（必要であれば）
git clone [https://github.com/b4mbo-o/image-bot.git](https://github.com/b4mbo-o/image-bot.git)
cd image-bot

# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

> **Note**: スクレイパーを使用する場合、`dlib` のビルドに必要なライブラリがシステムに入っている必要があります。
> 例 (Ubuntu): `sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev`

### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、Twitter APIキーを設定してください。

```ini
TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
```

### 3. 画像の準備

投稿したい画像は `images/` ディレクトリに配置してください。

---

## 🔖 ブランチと配布物

- `main`: もともとの全データ入りブランチ。
- `dev`: Bot/Scraperのコードと必要最低限のファイルのみ（画像・基準顔は空、`state/` は空ファイル）。
- `release-minimal.zip`: `bot.py`, `scraper.py`, `requirements.txt`, `README.md`, `.github/workflows/run.yml`, `.gitignore`, `state/` の最小セットをまとめた配布用ZIP。

---

## 🤖 使い方: 画像投稿 BOT

基本的なコマンド一覧です。

| 動作 | コマンド |
| --- | --- |
| **単発投稿** | `python bot.py` |
| **テキスト付き投稿** | `python bot.py --text "おはようございます"` |
| **常駐実行 (任意間隔)** | `python bot.py --loop --interval-hours 4` |
| **ドライラン (テスト)** | `python bot.py --dry-run` |

### オプション引数
- `--env-file`: `.env` の場所を変更する場合に指定。
- `--log-file`: ログをファイルに出力する場合に指定。
- `--history-size`: 履歴の保持件数を変更（デフォルト: 12）。

---

## 📷 使い方: 画像スクレイパー

Yahoo!リアルタイム検索から画像を収集し、顔認識でフィルタリングして保存します。

### 基本的な実行

```bash
# 基準顔: MEGAFON_noka/、デフォルトURLから直近2日分だけ取得
python scraper.py --out-dir images --reference-dir MEGAFON_noka --log-file logs/scrape.log
```

### 主なオプション

| オプション | 説明 |
| --- | --- |
| `--urls` | 取得元のYahoo検索URLを指定（複数指定可） |
| `--num-jitters` | 顔エンコードの試行回数。数値を上げると厳密になるが遅くなる（デフォルト: 5） |
| `--max-age-days` | 取得対象の日数（デフォルト: 2）。`0` を指定すると全期間対象 |
| `--tolerance` / `--negative-tolerance` / `--negative-margin` | 本人判定/NG判定のしきい値。デフォルトは顔マッチ0.50 / NG 0.40 / マージン0.05 |
| `--max-faces` | 1枚あたりの許容顔数（デフォルト: 2。3人以上はスキップ） |
| `--workers` | 顔エンコード並列数（デフォルト: 2） |
| `--html-file` | 保存済みHTMLファイルを解析してダウンロード（ローカル解析用） |

---

## ☁️ GitHub Actions 運用

- スケジュール: JST 00 / 06 / 08 / 12 / 16 / 20（UTC 15 / 21 / 23 / 03 / 07 / 11）
- 内容: `scraper.py` → `bot.py` を1ジョブで実行し、`images/`, `state/history.json`, `state/usage.json` の差分を自動コミット＆push（PR時はコミットしない）。
- 必須シークレット（リポや環境「.env」に設定）
  - `TWITTER_CONSUMER_KEY`
  - `TWITTER_CONSUMER_SECRET`
  - `TWITTER_ACCESS_TOKEN`
  - `TWITTER_ACCESS_TOKEN_SECRET`
  - 任意: `SAUCENAO_KEY`（逆引きAlt用）
- ログ: `logs/` を Artifact として保存。

※ Actionsで画像もリポにコミットされるため、リポサイズ増に注意。

---

## ⏰ ローカル/サーバーでの自動実行

ローカルやVPSで回す場合の例（systemdタイマーは現在停止中。必要なら以下を参考に再設定してください）。

### 投稿BOT (例: 4時間ごと)
```cron
0 */4 * * * cd /path/to/image-bot && /path/to/image-bot/.venv/bin/python bot.py --images-dir ./images --history-file ./state/history.json --log-file ./logs/cron.log >> ./logs/cron.log 2>&1
```

### スクレイパー (例: 毎日 AM3:00)
```cron
0 3 * * * cd /path/to/image-bot && /path/to/image-bot/.venv/bin/python scraper.py --log-file ./logs/scrape.log >> ./logs/scrape.log 2>&1
```

---

## ⚠️ 注意事項

- **スクレイパーの基準画像**: `images/` ディレクトリ（または `--reference-dir`）に基準となる顔画像がない場合、顔検出機能が正しく動作しません。
- **写真判定**: スクレイパーは「1〜2人の顔が写っている」かつ「基準顔と一致する」画像のみを保存します。
- **履歴管理**: 履歴ファイルはロック処理が行われるため、同時起動しても破損しにくい設計になっています。
