import asyncio
import uuid
import subprocess
import re
import argparse
import wave
from typing import Tuple, Optional

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

# キャッシュ（conditioning latents）生成に使う参照秒数
parser.add_argument(
    "--gpt_cond_len",
    type=int,
    default=30,
    help="Seconds used for GPT conditioning from reference audio. Default: 30",
)
parser.add_argument(
    "--max_ref_length",
    type=int,
    default=60,
    help="Max seconds used from reference audio. Default: 60",
)

args = parser.parse_args()

SPEAKER_WAV = args.voice
TEMPERATURE = args.temperature
TOP_P = args.top_p
REPETITION_PENALTY = args.repetition_penalty
SPEED = args.speed
HOST = args.host
PORT = args.port
GPT_COND_LEN = args.gpt_cond_len
MAX_REF_LENGTH = args.max_ref_length

# XTTSは24kHzで扱うことが多いので固定（必要なら変えてOK）
OUTPUT_SR = 24000

print("Loading XTTS model...")
# GPUが無い環境で落ちないようにする（cudaが無ければcpu）
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print("XTTS ready.")
print("Device:", device)
print("Using voice:", SPEAKER_WAV)
print(
    "Params:",
    f"temperature={TEMPERATURE}, top_p={TOP_P}, repetition_penalty={REPETITION_PENALTY}, speed={SPEED}",
)
print("Latents:", f"gpt_cond_len={GPT_COND_LEN}s, max_ref_length={MAX_REF_LENGTH}s")

# -----------------------------
# 音声キュー（重なり防止）
# -----------------------------
queue: asyncio.Queue[str] = asyncio.Queue()

# -----------------------------
# conditioning latents キャッシュ
# -----------------------------
xtts_model = None
gpt_cond_latent = None
speaker_embedding = None
use_cached_inference = False

try:
    # TTS APIの中の実モデル（XTTS）にアクセス
    xtts_model = tts.synthesizer.tts_model

    print("Building conditioning latents (cached)...")
    # TTS/XTTSの実装差を吸収するため、引数はなるべく名前付きで渡す
    # 成功すると (gpt_cond_latent, speaker_embedding) が得られる
    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=SPEAKER_WAV,
        gpt_cond_len=GPT_COND_LEN,
        max_ref_length=MAX_REF_LENGTH,
        sound_norm_refs=True,
    )

    use_cached_inference = True
    print("Latents cached OK. (Will use xtts_model.inference)")
except TypeError:
    # get_conditioning_latents の引数名が違う場合があるので、位置引数で再トライ
    try:
        print("Retrying conditioning latents with positional args...")
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            SPEAKER_WAV, GPT_COND_LEN, MAX_REF_LENGTH, True
        )
        use_cached_inference = True
        print("Latents cached OK. (Will use xtts_model.inference)")
    except Exception as e:
        print("[WARN] Could not cache latents (positional retry failed):", e)
        use_cached_inference = False
except Exception as e:
    print("[WARN] Could not cache latents:", e)
    use_cached_inference = False


def play_audio(path: str) -> None:
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
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


# 大文字英字の連続を日本語読みへ変換
def spell_out_acronym(match):
    mapping = {
        "A": "えー",
        "B": "びー",
        "C": "しー",
        "D": "でぃー",
        "E": "いー",
        "F": "えふ",
        "G": "じー",
        "H": "えいち",
        "I": "あい",
        "J": "じぇー",
        "K": "けー",
        "L": "える",
        "M": "えむ",
        "N": "えぬ",
        "O": "おー",
        "P": "ぴー",
        "Q": "きゅー",
        "R": "あーる",
        "S": "えす",
        "T": "てぃー",
        "U": "ゆー",
        "V": "ぶい",
        "W": "だぶりゅー",
        "X": "えっくす",
        "Y": "わい",
        "Z": "ぜっと",
    }

    word = match.group()
    return "".join(mapping.get(c, c) for c in word)


def katakana_to_hiragana(text: str) -> str:
    """
    カタカナをひらがなに変換する
    （全角カタカナのみ対応）
    """
    result = []
    for ch in text:
        code = ord(ch)
        # 全角カタカナ範囲
        if 0x30A1 <= code <= 0x30F6:
            result.append(chr(code - 0x60))
        else:
            result.append(ch)
    return "".join(result)


def normalize_text_for_tts(text: str):
    """
    表示用テキストと読み上げ用テキストを分けて返す
    return: (display_text, speak_text)
    """
    s = text.strip()

    # URLは読まない（必要なら「URL」とだけ読む）
    s = re.sub(r"https?://\S+", "、りんく。", s)

    # ｗ → ワラ
    s = re.sub(r"[ｗ]+", "ワラ", s)

    # 文末ワラは少し区切る
    s = re.sub(r"ワラ$", "、ワラ。", s)

    # カタカナをひらがなに
    s = katakana_to_hiragana(s)

    # 大文字英字を
    s = re.sub(r"\b[A-Z]{2,}\b", spell_out_acronym, s)

    # 数字を漢字へ変換
    def replace_number(match):
        try:
            return number2kanji(int(match.group()))
        except Exception:
            return match.group()

    s = re.sub(r"\d+", replace_number, s)
    
    # --- レス番号検出 ---
    # 例: レス350 → れす三百五十、
    m = re.match(r"^れす([一二三四五六七八九十百千万億〇零]+)", s)
    if m:
        number_part = m.group(1)
        # 読み上げ用
        speak_prefix = f"れす{number_part}、"
        # 表示用（レス番号削除）
        display_text = re.sub(r"^れす[一二三四五六七八九十百千万億〇零]+", "", s).strip()
        speak_text = speak_prefix + display_text
        return display_text, speak_text

    # レス番号が無い場合
    return s, s


def write_wav_int16(path: str, wav_float, sample_rate: int) -> None:
    """
    wav_float: 1D float array-like in [-1, 1]
    標準ライブラリだけでWAVを書き出す（soundfile不要）
    """
    import numpy as np

    if wav_float is None:
        raise ValueError("wav_float is None")

    # torch tensor -> numpy
    try:
        import torch  # type: ignore
        if isinstance(wav_float, torch.Tensor):
            wav_float = wav_float.detach().cpu().numpy()
    except Exception:
        pass

    wav_np = np.asarray(wav_float).reshape(-1)

    # float -> int16
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (wav_np * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def synthesize_to_file(text: str, out_path: str) -> None:
    """
    可能なら cached latents を使って推論。
    ダメなら tts.tts_to_file にフォールバック。
    """
    global use_cached_inference

    if use_cached_inference and xtts_model is not None and gpt_cond_latent is not None and speaker_embedding is not None:
        try:
            result = xtts_model.inference(
                text=text,
                language="ja",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                speed=SPEED,
            )
            wav = result["wav"] if isinstance(result, dict) and "wav" in result else result
            write_wav_int16(out_path, wav, OUTPUT_SR)
            return
        except Exception as e:
            print("[WARN] cached inference failed; fallback to tts_to_file:", e)
            use_cached_inference = False  # 以降は安定のためフォールバック固定

    # フォールバック（従来方式）
    tts.tts_to_file(
        text=text,
        speaker_wav=SPEAKER_WAV,
        language="ja",
        file_path=out_path,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        speed=SPEED,
    )


async def tts_worker():
    """
    キューを1件ずつ処理するワーカー
    """
    while True:
        text = await queue.get()
        try:
            output_path = f"/tmp/{uuid.uuid4()}.wav"

            # 音声生成（ブロック処理なのでスレッドに逃がす）
            await asyncio.to_thread(synthesize_to_file, text, output_path)

            # 再生もブロックなのでスレッドへ
            await asyncio.to_thread(play_audio, output_path)

        except Exception as e:
            print("TTS error:", e)

        finally:
            queue.task_done()


async def handler(websocket):
    async for message in websocket:
        text = extract_bouyomi_text(message)
        if not text:
            continue

        display_text, speak_text = normalize_text_for_tts(text)
        if not speak_text:
            continue

        # 🔹 ターミナルには本文だけ表示
        if display_text:
            print(display_text)

        # 🔹 読み上げはレス番号込み
        await queue.put(speak_text)


async def main():
    print(f"\n\nWebSocket待ち受け中...(ws://{HOST}:{PORT})")
    asyncio.create_task(tts_worker())
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # 永久待機


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n読み込み停止")

