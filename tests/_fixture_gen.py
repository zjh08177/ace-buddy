"""Generate fixture PNGs with known text for OCR tests.

Run manually: python tests/_fixture_gen.py
Outputs live in tests/fixtures/screens/.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCREENS = Path(__file__).parent / "fixtures" / "screens"


def make_text_png(path: Path, text: str, size: tuple[int, int] = (800, 400)) -> None:
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except OSError:
        font = ImageFont.load_default()
    # Multi-line rendering
    y = 40
    for line in text.splitlines():
        draw.text((40, y), line, fill=(0, 0, 0), font=font)
        y += 50
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")


def main() -> None:
    make_text_png(SCREENS / "hello.png", "hello ace buddy")
    make_text_png(
        SCREENS / "leetcode_two_sum.png",
        "Two Sum\nGiven an array of integers nums\nand an integer target, return\nindices of the two numbers that\nadd up to target.",
    )
    make_text_png(
        SCREENS / "behavioral_q.png",
        "Tell me about a time you\nhandled conflict on a team.",
    )
    print("Generated fixtures in", SCREENS)


if __name__ == "__main__":
    main()
