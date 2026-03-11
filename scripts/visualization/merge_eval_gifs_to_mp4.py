"""Merge evaluation GIFs under eval_logs into a single MP4 using MoviePy.

Ordering rule:
1. Read GIF filenames from the input directory.
2. Sort non-`hard` clips first.
3. Place clips whose stem contains `hard` at the end.
4. Within each group, use a natural filename sort (e.g. seed3 before seed12).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate repository root from script path.")
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


DEFAULT_INPUT_DIR = ROOT / "eval_logs"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "merged_eval_gifs.mp4"
_NATURAL_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GIF rollouts into a single MP4.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing GIF files. Default: eval_logs/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.gif",
        help="Glob pattern used to collect clips.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output MP4 frame rate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the ordered clip list without writing the MP4.",
    )
    return parser.parse_args()


def _natural_key(text: str) -> list[int | str]:
    parts = _NATURAL_RE.split(text.lower())
    key: list[int | str] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def collect_gifs(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    gifs = [path for path in input_dir.rglob(pattern) if path.is_file()]
    if not gifs:
        raise FileNotFoundError(f"No GIF files matching '{pattern}' were found under {input_dir}")
    return gifs


def sort_gifs(paths: list[Path]) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: (
            1 if "hard" in path.stem.lower() else 0,
            _natural_key(path.stem),
        ),
    )


def merge_gifs_to_mp4(paths: list[Path], output_path: Path, fps: int) -> None:
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError as exc:
        raise RuntimeError("moviepy is required. Install it with `pip install moviepy`.") from exc

    clips = []
    try:
        for path in paths:
            clip = VideoFileClip(str(path))
            clips.append(clip)
        merged = concatenate_videoclips(clips, method="compose")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_videofile(
            str(output_path),
            fps=max(1, int(fps)),
            codec="libx264",
            audio=False,
        )
        merged.close()
    finally:
        for clip in clips:
            clip.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    gifs = collect_gifs(input_dir, args.pattern)
    ordered = sort_gifs(gifs)

    print("[Merge] GIF order:")
    for idx, path in enumerate(ordered, start=1):
        print(f"  {idx:02d}. {path.name}")

    if args.dry_run:
        return

    merge_gifs_to_mp4(ordered, output_path, args.fps)
    print(f"[Merge] Saved MP4 to {output_path}")


if __name__ == "__main__":
    main()
