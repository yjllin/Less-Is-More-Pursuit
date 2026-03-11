"""Convert an MP4 rollout montage into a GIF for GitHub README display."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a video file into a GIF.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/merged_eval_gifs.mp4"),
        help="Input video path. Default: docs/merged_eval_gifs.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/merged_eval_gifs.gif"),
        help="Output GIF path. Default: docs/merged_eval_gifs.gif",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="GIF frame rate. Lower values keep file size manageable.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="Resize the GIF to the specified width in pixels.",
    )
    return parser.parse_args()


def convert_video_to_gif(input_path: Path, output_path: Path, fps: int, width: int | None) -> None:
    try:
        from moviepy import VideoFileClip
    except ImportError as exc:
        raise RuntimeError("moviepy is required. Install it with `pip install moviepy`.") from exc

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(input_path))
    resized_clip = clip
    try:
        if width and width > 0:
            try:
                resized_clip = clip.resized(width=width)
            except AttributeError:
                resized_clip = clip.resize(width=width)
        resized_clip.write_gif(str(output_path), fps=max(1, int(fps)))
    finally:
        if resized_clip is not clip:
            resized_clip.close()
        clip.close()


def main() -> None:
    args = parse_args()
    convert_video_to_gif(
        input_path=args.input.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        fps=args.fps,
        width=args.width,
    )
    print(f"[Convert] Saved GIF to {args.output}")


if __name__ == "__main__":
    main()
