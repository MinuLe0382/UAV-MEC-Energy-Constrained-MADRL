"""Combine v1 / v3 / v5 trajectory GIFs side-by-side into one comparison GIF."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SRC = {
    "v1\n(Lazy Agent\nF_uav=0.33)":   "results/v1/trajectory_maddpg.gif",
    "v3\n(Partial Fix\nF_uav=0.978)": "results/v3/trajectory_v3.gif",
    "v5\n(Solved\nF_uav=0.977)":      "results/presentation/v5_trajectory.gif",
    "v6\n(f=f_max 100%\nF_uav=0.88)": "results/presentation/v6_trajectory.gif",
}
OUT = Path("results/presentation/comparison_trajectory.gif")
TARGET_H = 400   # each panel height (px)
PAD       = 10   # padding between panels
LABEL_H   = 60   # height reserved for label above each panel
BORDER    = 4    # colored border thickness
FPS       = 4    # frames per second → duration = 1000/FPS ms

COLORS = {
    "v1\n(Lazy Agent\nF_uav=0.33)":   (220,  38,  38),  # red
    "v3\n(Partial Fix\nF_uav=0.978)": (234,  88,  12),  # orange
    "v5\n(Solved\nF_uav=0.977)":      ( 22, 163,  74),  # green
    "v6\n(f=f_max 100%\nF_uav=0.88)": ( 37,  99, 235),  # blue
}


def load_gif_frames(path: str) -> list[Image.Image]:
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(gif.copy().convert("RGB"))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def resize_to_height(img: Image.Image, h: int) -> Image.Image:
    ratio = h / img.height
    return img.resize((int(img.width * ratio), h), Image.LANCZOS)


def make_label_strip(text: str, w: int, h: int, color: tuple) -> Image.Image:
    strip = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw  = ImageDraw.Draw(strip)
    # Colored top bar
    draw.rectangle([0, 0, w, BORDER], fill=color)
    draw.rectangle([0, h - BORDER, w, h], fill=color)
    # Text (use default PIL font — avoids font path issues)
    lines = text.split("\n")
    line_h = (h - BORDER * 2 - 4) // max(len(lines), 1)
    for i, line in enumerate(lines):
        # measure text
        bbox = draw.textbbox((0, 0), line)
        tw = bbox[2] - bbox[0]
        tx = (w - tw) // 2
        ty = BORDER + 2 + i * line_h
        # shadow
        draw.text((tx + 1, ty + 1), line, fill=(180, 180, 180))
        draw.text((tx, ty), line, fill=color)
    return strip


def build_frame(panels: dict[str, Image.Image]) -> Image.Image:
    """Stack label + panel for each version, then join horizontally."""
    strips = []
    for label, img in panels.items():
        img_r = resize_to_height(img, TARGET_H)
        w = img_r.width
        color = COLORS[label]
        label_strip = make_label_strip(label, w, LABEL_H, color)
        # Add colored left/right borders
        bordered = Image.new("RGB", (w + BORDER * 2, TARGET_H), color=color)
        bordered.paste(img_r, (BORDER, 0))
        # Stack vertically
        combined = Image.new("RGB", (w + BORDER * 2, LABEL_H + TARGET_H),
                              color=(240, 240, 240))
        combined.paste(label_strip, (0, 0))
        combined.paste(bordered, (0, LABEL_H))
        strips.append(combined)

    # Horizontal join with padding
    total_w = sum(s.width for s in strips) + PAD * (len(strips) - 1)
    total_h = strips[0].height
    canvas = Image.new("RGB", (total_w, total_h), color=(240, 240, 240))
    x = 0
    for s in strips:
        canvas.paste(s, (x, 0))
        x += s.width + PAD
    return canvas


def main():
    print("Loading GIF frames...")
    all_frames: dict[str, list[Image.Image]] = {}
    for label, path in SRC.items():
        frames = load_gif_frames(path)
        print(f"  {path}: {len(frames)} frames")
        all_frames[label] = frames

    # Loop all to the same length (repeat shorter ones)
    max_len = max(len(f) for f in all_frames.values())
    for label in all_frames:
        f = all_frames[label]
        while len(f) < max_len:
            f.append(f[-1])

    print(f"Building {max_len} composite frames...")
    output_frames = []
    for i in range(max_len):
        panels = {label: all_frames[label][i] for label in all_frames}
        output_frames.append(build_frame(panels))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000 // FPS
    output_frames[0].save(
        OUT,
        save_all=True,
        append_images=output_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"[saved] {OUT}  ({len(output_frames)} frames, {duration_ms}ms/frame)")


if __name__ == "__main__":
    main()
