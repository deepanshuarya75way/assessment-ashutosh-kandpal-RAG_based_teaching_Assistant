"""Convert videos in videos/ to MP3 under audios/ using ffmpeg."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def parse_tutorial_parts(filename: str) -> tuple[str, str] | None:
    """
    Expected pattern (original Sigma-style): 'Title ｜ ... #NN [qualities].ext'
    Returns (tutorial_number, base_name_without_ext).
    """
    try:
        number = filename.split(" [")[0].split(" #")[1]
        base = filename.split(" ｜ ")[0]
        return number.strip(), base.strip()
    except (IndexError, ValueError):
        return None


def guess_name_fallback(path: Path) -> tuple[str, str]:
    """Use stem as title and '0' if pattern does not match."""
    return "0", path.stem


def main() -> None:
    root = Path(__file__).resolve().parent
    vid_dir = root / "videos"
    out_dir = root / "audios"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vid_dir.is_dir():
        print(f"Create folder and add videos: {vid_dir}", file=sys.stderr)
        sys.exit(1)

    exts = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpeg", ".mpg"}
    videos = sorted(
        p for p in vid_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not videos:
        print(f"No video files in {vid_dir}", file=sys.stderr)
        sys.exit(1)

    for path in videos:
        parsed = parse_tutorial_parts(path.name)
        if parsed:
            tutorial_number, file_name = parsed
        else:
            tutorial_number, file_name = guess_name_fallback(path)
            print(f"Warning: Non-standard filename, using #{tutorial_number} {file_name}: {path.name}")

        dst = out_dir / f"{tutorial_number}_{file_name}.mp3"
        print(path.name, "->", dst.name)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(dst)],
            cwd=str(root),
        )
        if r.returncode != 0:
            print(f"ffmpeg failed for {path}", file=sys.stderr)
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
