import socket
import threading
import uuid
import subprocess
from TTS.api import TTS

HOST = "127.0.0.1"
PORT = 50002

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("XTTS ready.")

def play_audio(path):
    subprocess.run(["ffplay", "-nodisp", "-autoexit", path],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

def handle_text(text):
    print("Received:", text)

    output_path = f"/tmp/{uuid.uuid4()}.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav="my_voice.wav",
        language="ja",
        file_path=output_path
    )

    play_audio(output_path)

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Listening on {HOST}:{PORT} ...")

        while True:
            conn, addr = server.accept()
            with conn:
                data = conn.recv(4096)
                if data:
                    try:
                        text = data.decode("utf-8", errors="ignore").strip()
                        if text:
                            threading.Thread(
                                target=handle_text,
                                args=(text,)
                            ).start()
                    except Exception as e:
                        print("Error:", e)

if __name__ == "__main__":
    start_server()

