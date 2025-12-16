# 4時間おきに画像を投稿するTwitter BOT

指定ディレクトリの画像を4時間ごとに投稿し、直近12回で使った画像は避けます（画像が足りない場合のみ再利用）。履歴は`state/history.json`に保存します。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Twitter APIキーを環境変数にセットしてください。

```bash
export TWITTER_CONSUMER_KEY=...
export TWITTER_CONSUMER_SECRET=...
export TWITTER_ACCESS_TOKEN=...
export TWITTER_ACCESS_TOKEN_SECRET=...
```

画像は`images/`以下（`.jpg/.jpeg/.png/.gif/.webp`）に配置してください。

## 使い方

- 単発で投稿: `python bot.py`
- 4時間ごとに常駐実行: `python bot.py --loop --interval-hours 4`
- テキスト付き: `python bot.py --text "コメント"`
- ドライラン（投稿せず選択のみ確認）: `python bot.py --dry-run`

履歴の保持件数は`--history-size`で変更できます（デフォルト12）。

## cronで回す例（推奨）

`crontab -e`で以下を追加すると4時間おきに1投稿します。

```
0 */4 * * * cd /root/image-bot && /root/image-bot/.venv/bin/python bot.py --images-dir /root/image-bot/images --history-file /root/image-bot/state/history.json --log-file /root/image-bot/logs/cron.log >> /root/image-bot/logs/cron.log 2>&1
```

`logs/`ディレクトリは自動作成されないので、必要なら事前に`mkdir -p /root/image-bot/logs`を実行してください。

## Yahoo!リアルタイム検索から画像を収集するスクリプト

`scraper.py`はYahoo!リアルタイム検索の結果ページを読み込み、`MEGAFON_noka/`（基準顔）にある画像と一致する写真のみ保存します。顔が3人以上いる集合写真は弾きます（ツーショまでは許可）。デフォルトで `https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_noka&aq=-1&ei=UTF-8&mtype=image&rkf=1` と `https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_idol&aq=-1&ei=UTF-8&mtype=image&rkf=1` から取得します。処理の最初に、`images/`内の完全重複（同一バイト列）を除外します。ページ内に新規画像が見つからなくなったら、そのソースの取得を終了します。
デフォルトは精度重視（tolerance 0.38, オレンジ救済なし）で、実行日から3日前までの投稿のみ対象にします。重複除外は完全一致（同一バイト列）のみです。

依存の`face_recognition`を使うため、初回はdlibビルド用のライブラリが必要です（例: `sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev`）。

実行例:

```bash
# 基準顔: MEGAFON_noka/、デフォルトURLから直近3日分だけ取得
python scraper.py --out-dir images --reference-dir MEGAFON_noka --log-file logs/scrape.log

# 取得元を変えたい場合（URLを並べて指定）
python scraper.py --urls https://search.yahoo.co.jp/realtime/search?p=ID%3AMEGAFON_noka\&aq=-1\&ei=UTF-8\&mtype=image\&rkf=1

# 追加オプション例
#   --num-jitters 3            # 顔エンコードをより厳密に（やや遅くなる）
#   --max-age-days 0           # 日付フィルタを無効化（全期間を対象にする）
#   --html-file get.html       # 保存済みHTMLを解析してダウンロード（ネットでページ取得しない）
#   --base-url https://search.yahoo.co.jp # --html-file時の相対URL基点（必要な場合のみ）
```

### 週1 cron 例

```
0 3 * * 0 cd /root/image-bot && /root/image-bot/.venv/bin/python scraper.py --log-file /root/image-bot/logs/scrape.log >> /root/image-bot/logs/scrape.log 2>&1
```

注意:
- 既存`images/`に顔が検出できない場合は保存されません（既存画像が基準データ）。
- 写真タイプ判定は1〜2人の顔が写っているものに限定します。2人写っていても基準顔と一致しない場合は保存しません。
- 同一ハッシュの画像は保存しません。

## 補足

- 画像が直近履歴と全て重複している場合のみ再利用します。
- 履歴ファイルのロックを行うので、同時起動しても履歴が壊れにくくなっています。
