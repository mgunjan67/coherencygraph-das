from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1] / "output" / "qa"
OUT = ROOT / "sheets"
OUT.mkdir(parents=True, exist_ok=True)

for group in ["main", "supplement", "cover"]:
    pages = sorted((ROOT / group).glob("*.png"))
    for start in range(0, len(pages), 4):
        chunk = pages[start : start + 4]
        opened = [Image.open(path).convert("RGB") for path in chunk]
        width = max(image.width for image in opened)
        height = max(image.height for image in opened)
        sheet = Image.new("RGB", (2 * width + 30, 2 * height + 30), "#d9d9d9")
        draw = ImageDraw.Draw(sheet)
        for offset, (path, page) in enumerate(zip(chunk, opened, strict=True)):
            x = (offset % 2) * (width + 20)
            y = (offset // 2) * (height + 20)
            sheet.paste(page, (x, y))
            draw.rectangle((x, y, x + 74, y + 20), fill="white")
            draw.text((x + 4, y + 3), path.stem, fill="black")
        sheet.save(OUT / f"{group}_{start // 4 + 1:02d}.png")
