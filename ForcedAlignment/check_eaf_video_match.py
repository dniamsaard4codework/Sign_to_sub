"""
check_eaf_video_match.py
------------------------
Verify that every EAF file in elan_forced_alignment/ has a corresponding
video file somewhere under ForcedAlignment/.

Reports TWO views of the dataset:

  [Part A] Stem-based lookup (functional pipeline test)
    -- The numeric filename of each EAF (e.g. "42.eaf") matches a video
       file with the same stem ("42.mp4") somewhere under ForcedAlignment/.
    -- This is what the pipeline actually uses; stem match = pipeline OK.

  [Part B] Deep MEDIA_DESCRIPTOR audit (data hygiene)
    -- Try to resolve RELATIVE_MEDIA_URL against the EAF's directory.
    -- Inspect MEDIA_URL (absolute) for local-vs-network paths.
    -- Verifies whether EAFs can load their videos in ELAN GUI on this machine.

Why deep audit matters even when stem-match is 100%:
  - Historically, EAFs were authored on multiple machines with different Thai encodings:
      * EAFs 1.eaf and 3.eaf use cp874 mojibake matching the on-disk folder.
      * Most other EAFs use proper UTF-8 Thai but without the space the
        on-disk folder name actually contains ("หนังสือ ภาษามือไทย" on disk
        vs "หนังสือภาษามือไทย" in EAF).
      * 240 EAFs (file 372 onward) have empty RELATIVE_MEDIA_URL entirely.
      * 301 EAFs reference a network host (file://192.168.1.18/...) only.
  - The pipeline ignores these by using stem-based lookup. Other tools
    (ELAN GUI, naive scripts) would fail before fix_eaf_media_paths.py repair.

Exits 0 if Part A passes (all stems match); 2 if Part A passes but Part B
has issues; 1 if any EAF is unrecoverable.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

FA_DIR  = Path(__file__).resolve().parent          # ForcedAlignment/
EAF_DIR = FA_DIR / "elan_forced_alignment"


def build_video_index(root: Path) -> dict[str, Path]:
    """Return {stem: path} for every .mp4 found anywhere under root."""
    return {p.stem: p for p in root.rglob("*.mp4")}


def extract_video_stem(eaf_path: Path) -> str | None:
    """
    Parse the EAF and return the video filename stem (without extension) from
    whichever URL field is populated, or None if neither is present.
    """
    try:
        root = ET.parse(eaf_path).getroot()
    except ET.ParseError:
        return None

    desc = root.find(".//MEDIA_DESCRIPTOR")
    if desc is None:
        return None

    for attr in ("RELATIVE_MEDIA_URL", "MEDIA_URL"):
        url = desc.get(attr, "").strip()
        if url:
            # Take the last path segment and strip extension
            filename = unquote(url.replace("\\", "/").split("/")[-1])
            stem = Path(filename).stem
            if stem:
                return stem

    return None


def deep_audit_one(eaf_path: Path) -> dict:
    """Inspect MEDIA_DESCRIPTOR of one EAF; return classification dict."""
    out = {
        "rel_status": "no_descriptor",  # ok | empty | broken | wrong_stem | no_descriptor
        "abs_status": "no_descriptor",  # empty | local | network | unknown | no_descriptor
        "rel_url": "",
        "abs_url": "",
    }
    try:
        root = ET.parse(eaf_path).getroot()
    except ET.ParseError:
        return out
    desc = root.find(".//MEDIA_DESCRIPTOR")
    if desc is None:
        return out

    rel = desc.get("RELATIVE_MEDIA_URL", "").strip()
    abs_url = desc.get("MEDIA_URL", "").strip()
    out["rel_url"] = rel
    out["abs_url"] = abs_url

    # RELATIVE_MEDIA_URL classification
    if not rel:
        out["rel_status"] = "empty"
    else:
        decoded = unquote(rel)
        if decoded.startswith("./"):
            decoded = decoded[2:]
        resolved = (eaf_path.parent / decoded).resolve()
        decoded_stem = Path(decoded).stem
        if not resolved.exists():
            out["rel_status"] = "broken"
        elif decoded_stem != eaf_path.stem:
            out["rel_status"] = "wrong_stem"
        else:
            out["rel_status"] = "ok"

    # MEDIA_URL classification
    if not abs_url:
        out["abs_status"] = "empty"
    elif abs_url.startswith("file://"):
        rest = abs_url[len("file://"):]
        if rest.startswith("/"):
            out["abs_status"] = "local"
        else:
            out["abs_status"] = "network"
    else:
        out["abs_status"] = "unknown"

    return out


def main() -> int:
    eaf_files = sorted(
        EAF_DIR.glob("*.eaf"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
    )
    if not eaf_files:
        print(f"No EAF files found in {EAF_DIR}")
        return 1

    print("Building video index ...", end=" ", flush=True)
    video_index = build_video_index(FA_DIR)
    print(f"{len(video_index)} videos found")

    # ── Part A: stem-based lookup ───────────────────────────────────────────
    no_descriptor: list[str] = []
    stem_mismatch: list[tuple[str, str]] = []
    not_found: list[tuple[str, str]] = []
    ok = 0

    for eaf in eaf_files:
        video_stem = extract_video_stem(eaf)
        if video_stem is None:
            no_descriptor.append(eaf.name)
            continue
        if video_stem != eaf.stem:
            stem_mismatch.append((eaf.name, video_stem + ".mp4"))
            continue
        if video_stem not in video_index:
            not_found.append((eaf.name, video_stem + ".mp4"))
            continue
        ok += 1

    total = len(eaf_files)
    print("\n" + "=" * 70)
    print(f"[Part A] Stem-based lookup (pipeline functional test) -- {total} EAFs")
    print("=" * 70)
    print(f"  [OK] EAF stem == video stem AND video file exists : {ok}")
    print(f"  [X]  No media descriptor                          : {len(no_descriptor)}")
    print(f"  [X]  Stem mismatch (EAF name != video name)       : {len(stem_mismatch)}")
    print(f"  [X]  Video not found in tree                      : {len(not_found)}")

    def show(label: str, items: list, fmt) -> None:
        if not items:
            return
        print(f"\n--- {label} ---")
        for entry in items[:20]:
            print(" ", fmt(entry))
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")

    show("No media descriptor", no_descriptor, lambda x: x)
    show("Stem mismatches", stem_mismatch, lambda x: f"{x[0]:<14}  ->  {x[1]}")
    show("Video not found", not_found, lambda x: f"{x[0]:<14}  ({x[1]} missing from tree)")

    part_a_pass = (not no_descriptor and not stem_mismatch and not not_found)

    # ── Part B: deep MEDIA_DESCRIPTOR audit ──────────────────────────────────
    rel_counts = {"ok": 0, "empty": 0, "broken": 0, "wrong_stem": 0, "no_descriptor": 0}
    abs_counts = {"local": 0, "network": 0, "empty": 0, "unknown": 0, "no_descriptor": 0}
    network_hosts: dict[str, int] = {}
    rel_broken_samples: list[tuple[str, str]] = []
    rel_empty_samples: list[str] = []

    for eaf in eaf_files:
        info = deep_audit_one(eaf)
        rel_counts[info["rel_status"]] += 1
        abs_counts[info["abs_status"]] += 1
        if info["abs_status"] == "network":
            host = info["abs_url"][len("file://"):].split("/")[0]
            network_hosts[host] = network_hosts.get(host, 0) + 1
        if info["rel_status"] == "broken" and len(rel_broken_samples) < 3:
            rel_broken_samples.append((eaf.name, info["rel_url"]))
        if info["rel_status"] == "empty" and len(rel_empty_samples) < 3:
            rel_empty_samples.append(eaf.name)

    print("\n" + "=" * 70)
    print(f"[Part B] Deep MEDIA_DESCRIPTOR audit -- {total} EAFs")
    print("=" * 70)
    print("  RELATIVE_MEDIA_URL:")
    print(f"    [OK] resolves to existing video, stem matches  : {rel_counts['ok']}")
    print(f"    [X]  empty                                     : {rel_counts['empty']}")
    print(f"    [X]  points to non-existent path               : {rel_counts['broken']}")
    print(f"    [WARN] resolves but stem mismatch              : {rel_counts['wrong_stem']}")
    print(f"    [-]  no MEDIA_DESCRIPTOR at all                : {rel_counts['no_descriptor']}")
    print("  MEDIA_URL (absolute):")
    print(f"    [OK] file:// local                             : {abs_counts['local']}")
    print(f"    [X]  file:// network (unreachable)             : {abs_counts['network']}")
    print(f"    [X]  empty                                     : {abs_counts['empty']}")
    print(f"    [X]  unknown scheme                            : {abs_counts['unknown']}")
    print(f"    [-]  no MEDIA_DESCRIPTOR                       : {abs_counts['no_descriptor']}")

    if network_hosts:
        print("\n  Network MEDIA_URL hosts (unreachable from this machine):")
        for host, count in sorted(network_hosts.items(), key=lambda x: -x[1]):
            print(f"    {host:<20} -> {count} EAFs")

    if rel_broken_samples:
        print("\n  Sample broken RELATIVE_MEDIA_URL entries:")
        for name, rel in rel_broken_samples:
            print(f"    {name}: {rel[:80]}{'...' if len(rel) > 80 else ''}")

    part_b_clean = (rel_counts["ok"] == total)

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if part_a_pass and part_b_clean:
        print("[OK] Both stem-based lookup AND MEDIA_DESCRIPTOR paths are clean.")
        return 0
    if part_a_pass and not part_b_clean:
        print("[OK] Pipeline can use stem-based lookup -- all 1132 clips reachable.")
        print("[WARN] MEDIA_DESCRIPTOR paths are unreliable; EAFs may fail to load")
        print("       in ELAN GUI without manual re-linking. See plan section 2.1.")
        return 2
    print("[X] Some EAFs cannot be resolved even by stem -- pipeline blocker.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
