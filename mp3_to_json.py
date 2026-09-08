"""Transcribe MP3 lessons with Whisper; write segmented JSON under jsons/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import whisper


def main() -> None:
    root = Path(__file__).resolve().parent
    audio_dir = root / "audios"
    json_dir = root / "jsons"
    json_dir.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="large-v2",
        help="Whisper model size (tiny/base/small/medium/large-v2)",
    )
    p.add_argument(
        "--language",
        default="en",
        help="ISO language code, or omit with --auto-lang for Whisper auto detect",
    )
    p.add_argument(
        "--auto-lang",
        action="store_true",
        help="Let Whisper detect language (ignores --language)",
    )
    p.add_argument(
        "--translate",
        action="store_true",
        help="Translate speech to English (Whisper translate task)",
    )
    args = p.parse_args()

    if not audio_dir.is_dir():
        print(f"Missing folder: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    model = whisper.load_model(args.model)
    task = "translate" if args.translate else "transcribe"

    for path in sorted(audio_dir.glob("*.mp3")):
        name = path.name
        if "_" not in name:
            print(f"Skip (expected NNN_title.mp3): {name}")
            continue
        number, stem = name.split("_", 1)
        title = stem[:-4] if stem.lower().endswith(".mp3") else stem
        print(number, title)

        transcribe_kw: dict = {
            "audio": str(path),
            "task": task,
            "word_timestamps": False,
        }
        if not args.auto_lang and args.language:
            transcribe_kw["language"] = args.language

        result = model.transcribe(**transcribe_kw)
        chunks = []
        for segment in result["segments"]:
            chunks.append(
                {
                    "number": number.strip(),
                    "title": title,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
            )
        out = {"chunks": chunks, "text": result["text"]}
        out_path = json_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
