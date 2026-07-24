"""Convert the source artwork into the app icon (black line art + alpha).

Takes a dark-lines-on-white image, crops it to the drawing, and writes a
square transparent PNG where alpha comes from darkness — so antialiased
edges survive and macOS can use it as a menu-bar template ("mask") icon.

Run with: uv run --with pillow python scripts/make_icon.py <source-image>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).parent.parent / "src" / "pywhispr" / "assets"
SIZE = 512
PADDING_FRACTION = 0.10
DARK_THRESHOLD = 100  # bbox detection: solid line-art pixels only
WATERMARK_SKIRT = 0.10  # ignore the bottom strip (stock-photo watermark text)
BINARY_THRESHOLD = 110  # luminance: below = line art, above = background/watermark
BUBBLE_ALPHA = 130  # speech bubble translucency (0-255)


def main(source: str) -> None:
    gray = np.asarray(Image.open(source).convert("L"), dtype=np.float32)
    height, width = gray.shape

    # Bounding box of the drawing, ignoring the watermark strip at the bottom.
    search = gray[: int(height * (1 - WATERMARK_SKIRT)), :]
    ys, xs = np.nonzero(search < DARK_THRESHOLD)
    if len(ys) == 0:
        raise SystemExit("No dark pixels found — is this the right image?")
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()

    # Square crop around the drawing, with padding.
    box_h, box_w = bottom - top, right - left
    side = int(max(box_h, box_w) * (1 + 2 * PADDING_FRACTION))
    cy, cx = (top + bottom) // 2, (left + right) // 2
    canvas = np.full((side, side), 255.0, dtype=np.float32)
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    crop = gray[y0 : min(height, y0 + side), x0 : min(width, x0 + side)]
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop

    # The stock-photo watermark text is mid-gray (luminance ~120-150), so a
    # smooth darkness→alpha ramp keeps a ghost of it. Instead: hard-threshold
    # to a binary mask (bold line art only), close small notches where the
    # watermark crossed a stroke, then blur slightly to restore antialiasing.
    mask = Image.fromarray(((canvas < BINARY_THRESHOLD) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))

    # Alamy also stamps translucent white text mid-image, cutting notches into
    # the strokes it crosses. A median filter wider than the text's stroke
    # width fills those in; applied only to the central band so it can't eat
    # thin lines elsewhere.
    healed = mask.filter(ImageFilter.MedianFilter(13))
    band = np.zeros((side, side), dtype=bool)
    band[int(side * 0.35) : int(side * 0.60), int(side * 0.20) : int(side * 0.80)] = True
    mask_arr = np.asarray(mask).copy()
    mask_arr[band] = np.asarray(healed)[band]

    mask = Image.fromarray(mask_arr).filter(ImageFilter.GaussianBlur(1.5))

    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[..., 3] = np.asarray(mask, dtype=np.uint8)

    art = Image.fromarray(rgba)

    out = ASSETS / "icon.png"
    compose_on_bubble(art).save(out)
    print(f"wrote {out} ({SIZE}x{SIZE}, from {source})")


def compose_on_bubble(art: Image.Image) -> Image.Image:
    """Place the line art inside a semi-transparent, black-outlined speech bubble."""
    s = SIZE * 2  # supersample for smooth edges, downscale at the end

    # Silhouette of the bubble: rounded body + tail pointing bottom-left.
    union = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(union)
    draw.rounded_rectangle((s * 0.03, s * 0.03, s * 0.97, s * 0.79), radius=s * 0.20, fill=255)
    draw.polygon([(s * 0.20, s * 0.75), (s * 0.13, s * 0.97), (s * 0.42, s * 0.79)], fill=255)

    # Outline = dilated silhouette minus silhouette, so it hugs the union of
    # both shapes with uniform width and matches the line-art style.
    dilated = union.filter(ImageFilter.MaxFilter(15))
    union_a = np.asarray(union, dtype=np.int16)
    ring_a = np.clip(np.asarray(dilated, dtype=np.int16) - union_a, 0, 255)

    outline_layer = np.zeros((s, s, 4), dtype=np.uint8)
    outline_layer[..., 3] = ring_a.astype(np.uint8)
    fill_layer = np.zeros((s, s, 4), dtype=np.uint8)
    fill_layer[..., :3] = 255
    fill_layer[..., 3] = (union_a * BUBBLE_ALPHA // 255).astype(np.uint8)

    bubble = Image.fromarray(outline_layer)
    bubble.alpha_composite(Image.fromarray(fill_layer))

    # Art centered in the bubble body.
    inner = int(s * 0.64)
    scaled = art.resize((inner, inner), Image.LANCZOS)
    bubble.alpha_composite(scaled, (int(s * 0.5 - inner / 2), int(s * 0.41 - inner / 2)))

    return bubble.resize((SIZE, SIZE), Image.LANCZOS)


if __name__ == "__main__":
    main(sys.argv[1])
