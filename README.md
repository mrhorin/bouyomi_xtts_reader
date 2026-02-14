# XTTS WebSocket Reader

jpnkn掲示板（棒読みちゃん互換 WebSocket）から送られてくるテキストを  
XTTS (xtts_v2) を使って好きな声で読み上げるローカルTTSサーバー

---

## 動作環境

- Ubuntu / WSL2 推奨
- Python 3.10 推奨
- NVIDIA GPU + CUDA（任意）
- ffmpeg / MeCab 必須

---

## 初回セットアップ

プロジェクトフォルダへ移動
```bash
cd your_project_directory
```

仮想環境を作成
```bash
python3 -m venv venv
```

仮想環境を有効化
```bash
source venv/bin/activate
```

pipパッケージをインストール
```bash
pip install -r requirements.txt
```

OS側パッケージをインストール（WSL / Ubuntu）
```bash
sudo apt update
sudo apt install -y ffmpeg mecab libmecab-dev mecab-ipadic-utf8
```

## 使い方

サンプル音声ファイルへのパスを指定し、スクリプトを実行すると`ws://localhost:50002`で listen する
```bash
python reader_jpkn.py --voice your_voice.wav
```

## オプション

```bash
usage: reader_jpnkn.py [-h] --voice VOICE [--temperature TEMPERATURE] [--host HOST]
                       [--port PORT]

WebSocket Bouyomi-compatible TTS Reader (XTTS)

options:
  -h, --help            show this help message and exit
  --voice VOICE         Path to speaker voice file (wav/mp3/etc). Recommended: wav.
  --temperature TEMPERATURE
                        TTS temperature (default: 0.9)
  --host HOST           Listen host (default: 127.0.0.1)
  --port PORT           Listen port (default: 50002)
```

