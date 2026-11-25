"""
Collect openly licensed road-traffic photos of Indian cities from Wikimedia Commons, as a test set from
a source the detector never saw in training.

    python scripts/collect_commons.py --list          # list candidates (title, size, licence) only
    python scripts/collect_commons.py                 # download into data/own_photos/raw/<place>/
    python scripts/collect_commons.py --category "Category:Streets in Bengaluru" --depth 1 --only keep.txt

Default start: https://commons.wikimedia.org/wiki/Category:Traffic_congestion_in_India and its
sub-categories (one per city/state; the sub-category becomes the <place> folder, which keeps the split
grouped by place). That category is almost only jams, so ordinary street categories ("Streets in
Bengaluru", "Streets in Delhi", ...) were added for free-flowing and moderate traffic; --only restricts a
run to the titles kept after looking at every candidate.
Only freely licensed files are kept (CC0, public domain, CC BY, CC BY-SA). Images are downloaded at most
1920 px wide into raw/<place>/; resized copies (EXIF rotation applied, longer side <= 1920 px) go to
images/<place>/, which is what the notebooks read. Every file's title, author, licence and page URL is written to
data/own_photos/commons_sources.csv, which is the attribution required by these licences.
Requests follow the Wikimedia API etiquette (one request at a time, descriptive User-Agent, short pauses).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "EduTrafficDatasetBuilder/1.0 (student computer-vision project)"}
START = "Category:Traffic congestion in India"
OK_LICENCE = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? \d)", re.I)


def api(**params):
    params.update(format="json", formatversion=2)
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    for attempt in range(6):
        time.sleep(1.5)                                   # stay well under the API rate limit
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:              # 429: wait as long as the server asks
            wait = int(e.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
            print(f"   HTTP {e.code}, waiting {wait} s")
            time.sleep(wait)
        except Exception as e:  # transient network error
            print("   retry", attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Wikimedia API not reachable")


def members(cat, kind):
    out, cont = [], {}
    while True:
        r = api(action="query", list="categorymembers", cmtitle=cat, cmtype=kind, cmlimit=500, **cont)
        out += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" not in r:
            return out
        cont = r["continue"]


def walk(cat, place, depth, seen, files):
    if cat in seen or depth < 0:
        return
    seen.add(cat)
    for f in members(cat, "file"):
        files.setdefault(f, place)
    for sub in members(cat, "subcat"):
        sub_place = place if place != "india" else re.sub(r"[^a-z0-9]+", "_",
                                                           sub.split(" in ")[-1].lower()).strip("_")
        walk(sub, sub_place, depth - 1, seen, files)


def plain(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def place_of(cat):
    return "india" if cat == START else re.sub(r"[^a-z0-9]+", "_", cat.split(" in ")[-1].lower()).strip("_")


def info(titles, width=1920):
    r = api(action="query", prop="imageinfo", titles="|".join(titles), iiprop="url|size|mime|extmetadata",
            iiurlwidth=width)
    out = {}
    for p in r["query"]["pages"]:
        if "imageinfo" not in p:
            continue
        ii = p["imageinfo"][0]
        md = ii.get("extmetadata", {})
        out[p["title"]] = {
            "width": ii["width"], "height": ii["height"], "mime": ii["mime"],
            "download_url": ii.get("thumburl") or ii["url"], "page_url": ii["descriptionurl"],
            "licence": plain(md.get("LicenseShortName", {}).get("value")),
            "licence_url": plain(md.get("LicenseUrl", {}).get("value")),
            "author": plain(md.get("Artist", {}).get("value"))[:150],
            "date": plain(md.get("DateTimeOriginal", {}).get("value"))[:30],
        }
    return out


def normalise_all(max_side=1920):
    """raw/<place>/*.jpg -> images/<place>/*.jpg: EXIF rotation applied, RGB, longer side <= max_side."""
    from PIL import Image, ImageOps
    own = ROOT / "data" / "own_photos"
    n = 0
    for src in sorted((own / "raw").rglob("*.jpg")):
        dst = own / "images" / src.relative_to(own / "raw")
        if dst.exists():
            continue
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            s = max_side / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=92)
        n += 1
    print(f"normalised {n} new photo(s) into {(own / 'images').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="only list the candidates")
    ap.add_argument("--min-width", type=int, default=800)
    ap.add_argument("--skip", default="", help="comma-separated file titles to leave out (not road traffic)")
    ap.add_argument("--category", action="append",
                    help=f"Commons category to walk (repeatable; default '{START}')")
    ap.add_argument("--depth", type=int, default=3, help="how many levels of sub-categories to follow")
    ap.add_argument("--only", help="text file with one 'File:...' title per line: download just these "
                                   "(the photos kept after visual screening)")
    ap.add_argument("--place", help="with --only: <place> folder for all listed titles (skips the category walk)")
    ap.add_argument("--normalise-only", action="store_true",
                    help="only (re)create the resized copies in data/own_photos/images/")
    args = ap.parse_args()
    if args.normalise_only:
        normalise_all()
        return

    files: dict[str, str] = {}
    if args.only and args.place:          # titles already known: no category walk needed
        files = {ln.strip(): args.place for ln in Path(args.only).read_text(encoding="utf-8").splitlines()
                 if ln.strip()}
    else:
        seen: set[str] = set()
        for cat in args.category or [START]:
            walk(cat, place_of(cat), depth=args.depth, seen=seen, files=files)
    titles = [t for t in files if re.search(r"\.(jpe?g|png)$", t, re.I)]
    if args.only:
        wanted = {ln.strip() for ln in Path(args.only).read_text(encoding="utf-8").splitlines() if ln.strip()}
        titles = [t for t in titles if t in wanted]
    meta = {}
    for i in range(0, len(titles), 40):
        meta.update(info(titles[i:i + 40]))
        time.sleep(0.5)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    keep = [t for t in titles if t in meta and t not in skip and meta[t]["width"] >= args.min_width
            and OK_LICENCE.match(meta[t]["licence"])]
    print(f"{len(files)} files in the category tree, {len(titles)} JPEG/PNG, {len(keep)} freely licensed and "
          f">= {args.min_width} px wide")
    if args.list:
        for t in keep:
            m = meta[t]
            print(f"  [{files[t]}] {t}  {m['width']}x{m['height']}  {m['licence']}")
        return

    raw = ROOT / "data" / "own_photos" / "raw"
    rows, used = [], set()
    for t in keep:
        m, place = meta[t], files[t]
        slug = (re.sub(r"[^A-Za-z0-9]+", "_", t.removeprefix("File:").rsplit(".", 1)[0]).strip("_")
                or "commons_" + hashlib.md5(t.encode("utf-8")).hexdigest()[:10])   # non-Latin title
        name = (slug if len(slug) <= 80 else slug[:70] + "_" + slug[-9:]) + ".jpg"   # keep the tail: burst
        while f"{place}/{name}" in used:                                          # shots differ only there
            name = name[:-4] + "_b.jpg"
        used.add(f"{place}/{name}")
        dest = raw / place / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            if dest.exists():
                break
            try:
                with urllib.request.urlopen(urllib.request.Request(m["download_url"], headers=UA), timeout=120) as r:
                    dest.write_bytes(r.read())
                time.sleep(1.5)
            except urllib.error.HTTPError as e:
                wait = int(e.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
                print(f"   HTTP {e.code} on download, waiting {wait} s")
                time.sleep(wait)
            except Exception as e:  # dropped connection, DNS hiccup
                print(f"   network error ({e}), retrying")
                time.sleep(10 * (attempt + 1))
        if not dest.exists():
            print("   skipped (download failed):", t)
            continue
        rows.append({"file": f"{place}/{name}", "place": place, "commons_title": t, "author": m["author"],
                     "licence": m["licence"], "licence_url": m["licence_url"], "date": m["date"],
                     "page_url": m["page_url"]})
        print(f"  {place}/{name}")
    # merge with earlier runs (other categories); drop rows that were skipped or whose file is gone
    out = ROOT / "data" / "own_photos" / "commons_sources.csv"
    merged = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            merged = {r["commons_title"]: r for r in csv.DictReader(f)}
    merged.update({r["commons_title"]: r for r in rows})
    final = [r for t, r in merged.items() if t not in skip and (raw / r["file"]).exists()]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(final[0]))
        w.writeheader()
        w.writerows(sorted(final, key=lambda r: r["file"]))
    print(f"downloaded {len(rows)} images in this run; {len(final)} in total; "
          f"attribution -> {out.relative_to(ROOT)}")
    normalise_all()


if __name__ == "__main__":
    main()
