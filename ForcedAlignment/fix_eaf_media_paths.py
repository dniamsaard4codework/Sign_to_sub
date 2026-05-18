"""
Repair ELAN media links for the ForcedAlignment dataset.

Each EAF in elan_forced_alignment/ is named with the same numeric stem as its
matching MP4 somewhere below ForcedAlignment/. This script rewrites the EAF
MEDIA_DESCRIPTOR so ELAN can resolve the correct local video directly.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FA_DIR = Path(__file__).resolve().parent
EAF_DIR = FA_DIR / "elan_forced_alignment"

ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def build_video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for video in root.rglob("*.mp4"):
        if video.stem in index:
            duplicates.setdefault(video.stem, [index[video.stem]]).append(video)
        else:
            index[video.stem] = video.resolve()

    if duplicates:
        print("Duplicate video stems found; refusing to guess:")
        for stem, paths in sorted(duplicates.items())[:20]:
            print(f"  {stem}:")
            for path in paths:
                print(f"    {path}")
        return {}

    return index


def relative_media_url(eaf_path: Path, video_path: Path) -> str:
    rel_path = os.path.relpath(video_path.resolve(), eaf_path.parent.resolve())
    return Path(rel_path).as_posix()


def media_url(video_path: Path) -> str:
    # ELAN handles these dataset paths more reliably as literal local file URLs
    # than as percent-encoded URLs because the folder names are mojibake text.
    return "file:///" + video_path.resolve().as_posix()


def repair_one(eaf_path: Path, video_path: Path) -> bool:
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    descriptors = root.findall(".//MEDIA_DESCRIPTOR")
    if len(descriptors) != 1:
        raise ValueError(f"{eaf_path.name}: expected 1 MEDIA_DESCRIPTOR, found {len(descriptors)}")

    descriptor = descriptors[0]
    new_media_url = media_url(video_path)
    new_relative_url = relative_media_url(eaf_path, video_path)

    old_values = (
        descriptor.get("MEDIA_URL", ""),
        descriptor.get("RELATIVE_MEDIA_URL", ""),
        descriptor.get("MIME_TYPE", ""),
    )
    new_values = (new_media_url, new_relative_url, "video/mp4")

    if old_values == new_values:
        return False

    descriptor.set("MEDIA_URL", new_media_url)
    descriptor.set("RELATIVE_MEDIA_URL", new_relative_url)
    descriptor.set("MIME_TYPE", "video/mp4")
    tree.write(eaf_path, encoding="utf-8", xml_declaration=True)
    return True


def main() -> int:
    eaf_files = sorted(EAF_DIR.glob("*.eaf"), key=numeric_sort_key)
    if not eaf_files:
        print(f"No EAF files found in {EAF_DIR}")
        return 1

    video_index = build_video_index(FA_DIR)
    if not video_index:
        return 1

    missing = [eaf.name for eaf in eaf_files if eaf.stem not in video_index]
    if missing:
        print(f"Missing matching MP4 for {len(missing)} EAF files:")
        for name in missing[:20]:
            print(f"  {name}")
        return 1

    updated = 0
    unchanged = 0
    for eaf_path in eaf_files:
        if repair_one(eaf_path, video_index[eaf_path.stem]):
            updated += 1
        else:
            unchanged += 1

    print(f"Checked {len(eaf_files)} EAF files")
    print(f"Updated {updated}")
    print(f"Already correct {unchanged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
