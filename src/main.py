import asyncio
import uuid
import subprocess
import re
import argparse

from TTS.api import TTS
import websockets
from kanjize import number2kanji

# -----------------------------
# コマンドライン引数
# -----------------------------
parser = argparse.ArgumentParser(description="WebSocket Bouyomi-compatible TTS Reader (XTTS)")

parser.add_argument(
    "--voice",
    type=str,
    required=True,
    help="Path to speaker reference audio (wav/mp3/etc). Recommended: wav.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.9,
    help="Sampling temperature (higher = more variation). Default: 0.9",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.9,
    help="Nucleus sampling top_p (lower = more stable). Default: 0.9",
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.1,
    help="Penalty for repetition (higher = less repetition). Default: 1.1",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Speech speed (1.0 = normal). Default: 1.0",
)
parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Listen host. Default: 127.0.0.1",
)
parser.add_argument(
    "--port",
    type=int,
    default=50002,
    help="Listen port. Default: 50002",
)

args = parser.parse_args()

SPEAKER_WAV = args.voice
TEMPERATURE = args.temperature
TOP_P = args.top_p
REPETITION_PENALTY = args.repetition_penalty
SPEED = args.speed
HOST = args.host
PORT = args.port

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("XTTS ready.")
print("Using voice:", SPEAKER_WAV)
print("Params:", f"temperature={TEMPERATURE}, top_p={TOP_P}, repetition_penalty={REPETITION_PENALTY}, speed={SPEED}")

# 音声キュー（重なり防止）
queue = asyncio.Queue()


def play_audio(path: str) -> None:
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def extract_bouyomi_text(message: str) -> str:
    """
    掲示板から届く <bouyomi> 区切りのメタ情報を剥がして本文だけ抽出する
    """
    s = message.strip()

    # <bouyomi>形式の除去：最後のセクションが本文になっているケースを想定
    if "<bouyomi>" in s:
        parts = [p for p in s.split("<bouyomi>") if p.strip()]
        s = parts[-1].strip() if parts else s

    # 先頭の大量数字（例: 1100100550）削除
    s = re.sub(r"^\d{3,}", "", s).strip()

    return s


def normalize_text_for_tts(text: str) -> str:
    s = text

    # ｗ → ワラ
    s = re.sub(r"[ｗw]+", "ワラ", s)

    # レス→ れす
    s = re.sub(r"レス(?=\d)", "れす", s)

    # 数字を漢字へ変換
    def replace_number(match):
        try:
            return number2kanji(int(match.group()))
        except Exception:
            return match.group()

    s = re.sub(r"\d+", replace_number, s)

    # 文末ワラは少し区切る
    s = re.sub(r"ワラ$", "ワラ。", s)

    return s.strip()


async def tts_worker():
    """
    キューを1件ずつ処理するワーカー
    """
    while True:
        text = await queue.get()
        try:
            print("Speaking:", text)

            output_path = f"/tmp/{uuid.uuid4()}.wav"

            # 音声生成
            await asyncio.to_thread(
                tts.tts_to_file,
                text=text,
                speaker_wav=SPEAKER_WAV,
                language="ja",
                file_path=output_path,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                speed=SPEED,
            )

            # 再生もブロックなのでスレッドへ
            await asyncio.to_thread(play_audio, output_path)

        except Exception as e:
            print("TTS error:", e)

        finally:
            queue.task_done()


async def handler(websocket):
    print("WebSocket connected")
    async for message in websocket:
        text = extract_bouyomi_text(message)
        if not text:
            continue

        text = normalize_text_for_tts(text)
        if not text:
            continue

        print("Queued:", text)
        await queue.put(text)


async def main():
    print(f"Listening WebSocket on ws://{HOST}:{PORT}")
    asyncio.create_task(tts_worker())
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # 永久待機


if __name__ == "__main__":
    asyncio.run(main())

