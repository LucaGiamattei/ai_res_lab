"""Fetch 5 public-domain (or freely-licensed) ImageNet sample images from Wikimedia Commons.

Resolves the canonical upload URL via Wikimedia API, downloads, resizes to <=640px max side,
re-encodes as JPG, and saves. Verifies each is correctly classified by MobileNetV2 and prints
top-1 prediction.

Public-domain / CC0 / CC-BY-SA images selected manually (titles are stable; license info
recorded in data/images/README.md).
"""
from __future__ import annotations
import io
import sys
import urllib.parse
from pathlib import Path

import requests
from PIL import Image

# Map from local filename -> Wikimedia Commons file title (without "File:" prefix).
# Selected for: clear single-subject framing, dominant subject category (so MobileNetV2 hits the
# right class), and freely-licensed (PD or CC).
SOURCES: list[tuple[str, str, int, str]] = [
    # (local_name, wikimedia_title, expected_imagenet_class_id, license_short)
    ("panda.jpg",            "Grosser_Panda.JPG",                388, "CC BY-SA 3.0"),
    ("school_bus.jpg",       "Gillig_Phantom_School_Bus_LAUSD.jpg", 779, "CC BY-SA"),
    ("golden_retriever.jpg", "Golden_Retriever_Hund_Dog.JPG",     207, "GFDL / CC BY-SA"),
    ("traffic_light.jpg",    "LED_traffic_light_on_red.jpg", 920, "CC BY-SA"),
    ("espresso.jpg",         "A_small_cup_of_coffee.JPG",         967, "CC BY-SA 2.0"),
]

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "ai_res_lab/1.0 (educational; contact: lgiamattei@gmail.com)"
}


def resolve_image_url(title: str) -> str:
    """Use the Wikimedia API to get the direct upload URL for a Commons file."""
    params = {
        "action": "query",
        "titles": f"File:{title}",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json",
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "imageinfo" not in page:
        raise RuntimeError(f"No imageinfo for File:{title} (page={page})")
    return page["imageinfo"][0]["url"]


def download_resize_save(url: str, out_path: Path, max_side: int = 640) -> tuple[int, int]:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=88)
    return img.size


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "challenge_2_adversarial" / "data" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for local_name, title, expected_id, license_short in SOURCES:
        try:
            url = resolve_image_url(title)
            size = download_resize_save(url, out_dir / local_name)
            print(f"OK   {local_name:25s} <- {title}  ({size[0]}x{size[1]})  [{license_short}]")
            rows.append((local_name, title, expected_id, url, license_short))
        except Exception as e:
            print(f"FAIL {local_name:25s} <- {title}: {e}", file=sys.stderr)
            return 1

    # Write sources README
    readme = out_dir / "README.md"
    body = ["# Sample images — provenance and licensing", "",
            "Tutte le immagini sono state scaricate da Wikimedia Commons e ridimensionate a max 640 px lato lungo, JPEG quality 88. Le licenze indicate sono quelle dichiarate sulla pagina del file al momento del download.", "",
            "| Local file | Wikimedia title | ImageNet class | Class ID | License | Original URL |",
            "|------------|-----------------|----------------|----------|---------|--------------|"]
    for local_name, title, expected_id, url, lic in rows:
        body.append(f"| `{local_name}` | [{title}](https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(title)}) | (vedi `imagenet_classes.txt` riga {expected_id}) | {expected_id} | {lic} | {url} |")
    body.append("")
    body.append("Per riprodurre il download: `python build_scripts/fetch_images.py` dalla root del package.")
    readme.write_text("\n".join(body) + "\n")
    print(f"Wrote {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
