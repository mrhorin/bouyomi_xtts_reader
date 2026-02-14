import asyncio
import uuid
import subprocess
import re
import argparse
from TTS.api import TTS
import websockets

# -----------------------------
# コマンドライン引数
# -----------------------------
parser = argparse.ArgumentParser(description="WebSocket TTS Server")
parser.add_argument(
    "--voice",
    type=str,
    required=True,
    help="Path to speaker wav file",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.9,
    help="TTS temperature (default: 0.9)",
)
args = parser.parse_args()

SPEAKER_WAV = args.voice
TEMPERATURE = args.temperature

HOST = "127.0.0.1"
PORT = 50002

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("XTTS ready.")
print("Using voice:", SPEAKER_WAV)
print("Temperature:", TEMPERATURE)

queue = asyncio.Queue()


def play_audio(path):
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def extract_bouyomi_text(message: str) -> str:
    s = message.strip()

    if "<bouyomi>" in s:
        parts = [p for p in s.split("<bouyomi>") if p.strip()]
        s = parts[-1].strip()

    s = re.sub(r"^\d{3,}", "", s).strip()
    s = re.sub(r"^レス\d+\s*\n?", "", s).strip()

    return s


async def tts_worker():
    while True:
        text = await queue.get()

        try:
            print("Speaking:", text)

            output_path = f"/tmp/{uuid.uuid4()}.wav"

            await asyncio.to_thread(
                tts.tts_to_file,
                text=text,
                speaker_wav=SPEAKER_WAV,
                language="ja",
                file_path=output_path,
                temperature=TEMPERATURE,
            )

            await asyncio.to_thread(play_audio, output_path)

        except Exception as e:
            print("TTS error:", e)

        finally:
            queue.task_done()


async def handler(websocket):
    print("WebSocket connected")

    async for message in websocket:
        text = extract_bouyomi_text(message)
        if text:
            print("Queued:", text)
            await queue.put(text)


async def main():
    print(f"Listening WebSocket on ws://{HOST}:{PORT}")

    asyncio.create_task(tts_worker())

    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

