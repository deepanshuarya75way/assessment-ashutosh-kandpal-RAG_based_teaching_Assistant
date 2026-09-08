"""
One entry point for the teaching-assistant data pipeline.

  python run_pipeline.py convert    # videos/*.mp4|* -> audios/*.mp3 (ffmpeg)
  python run_pipeline.py transcribe # audios/*.mp3 -> jsons/*.json (Whisper)
  python run_pipeline.py embed      # jsons/*.json -> embeddings.joblib (Ollama bge-m3)
  python run_pipeline.py ui         # open Streamlit app (Teaching assistant)
  python run_pipeline.py all        # convert + transcribe + embed (stops on first failure)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Teaching assistant ingestion pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("convert", help="Run video_to_mp3 (needs ffmpeg)")
    sub.add_parser("transcribe", help="Run mp3_to_json (Whisper)")
    sub.add_parser("embed", help="Run preprocess_json (Ollama embeddings)")
    sub.add_parser("ui", help="Start Streamlit app (app.py)")
    sub.add_parser("all", help="convert, then transcribe, then embed")

    args = p.parse_args()

    if args.cmd == "convert":
        run([sys.executable, str(ROOT / "video_to_mp3.py")], ROOT)
    elif args.cmd == "transcribe":
        run([sys.executable, str(ROOT / "mp3_to_json.py")], ROOT)
    elif args.cmd == "embed":
        run([sys.executable, str(ROOT / "preprocess_json.py")], ROOT)
    elif args.cmd == "ui":
        run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--browser.gatherUsageStats", "false"], ROOT)
    elif args.cmd == "all":
        run([sys.executable, str(ROOT / "video_to_mp3.py")], ROOT)
        run([sys.executable, str(ROOT / "mp3_to_json.py")], ROOT)
        run([sys.executable, str(ROOT / "preprocess_json.py")], ROOT)


if __name__ == "__main__":
    main()
