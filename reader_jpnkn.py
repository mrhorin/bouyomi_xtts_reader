import asyncio
import uuid
import subprocess
import re
from TTS.api import TTS
import websockets

HOST = "127.0.0.1"
PORT = 50002

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("XTTS ready.")

SPEAKER_WAV = "my_voice.wav"

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

            # 音声生成（ブロック処理なのでスレッドに逃がす）
            await asyncio.to_thread(
                tts.tts_to_file,
                text=text,
                speaker_wav=SPEAKER_WAV,
                language="ja",
                file_path=output_path,
                temperature=1.1,
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

