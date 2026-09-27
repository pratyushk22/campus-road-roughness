"""
Vision-model step: classify each defect photo into a defect type and severity.

    pip install anthropic
    export ANTHROPIC_API_KEY=...        # your own key
    python src/classify_photos.py                   # label every photo
    python src/classify_photos.py --dry-run         # show the prompt, no API calls
    python src/classify_photos.py --model claude-opus-5-5

Reads  data/raw/photos/*.jpg and data/processed/photo_locations.csv
Writes data/processed/photo_labels_api.csv  (label_source = inferred)

Note: the labels shipped in data/processed/photo_labels.csv were produced during the
event by Claude through the chat interface (free-text defect type + the same
low/medium/high severity scale). This script makes that step reproducible via the API
and uses a fixed category list so labels are comparable across runs.
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PHOTOS = BASE / "data" / "raw" / "photos"
PROCESSED = BASE / "data" / "processed"

CATEGORIES = [
    "pothole", "alligator_cracking", "linear_or_transverse_crack", "raveling_surface_wear",
    "surface_delamination", "failing_patch", "broken_sunken_or_missing_pavers",
    "slab_joint_or_edge_step", "paver_asphalt_joint_gap", "waterlogging_drainage",
    "speed_breaker", "manhole_or_cover", "no_visible_defect",
]

PROMPT = f"""You are inspecting a photo of a road surface on a university campus, taken by a
cyclist at a spot where a phone accelerometer recorded a bump.

Classify the MAIN defect visible. Use exactly one primary category from:
{", ".join(CATEGORIES)}
Optionally list other visible categories as secondary.

Severity for a cyclist:
- low: cosmetic or minor, rideable without slowing
- medium: noticeable jolt or risk; should be repaired within months
- high: structural failure, sharp edge, missing material or a hazard likely to cause a fall

Also say whether any person is identifiable in the photo.

Reply with ONLY a JSON object, no other text:
{{"primary": "<category>", "secondary": ["<category>", ...], "severity": "low|medium|high",
  "description": "<one sentence>", "person_identifiable": true|false, "confidence": 0.0-1.0}}"""


def classify(client, model, path):
    data = base64.standard_b64encode(path.read_bytes()).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    text = re.sub(r"```(json)?", "", text).strip()
    out = json.loads(text)
    if out.get("primary") not in CATEGORIES:
        out["primary"] = f"UNRECOGNISED:{out.get('primary')}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    photos = sorted(PHOTOS.glob("*.jpg"))
    print(f"{len(photos)} photos in {PHOTOS}")
    if a.dry_run:
        print(PROMPT)
        return
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

    rows = []
    for p in photos:
        try:
            r = classify(client, a.model, p)
        except Exception as e:                # keep going; record the failure
            r = {"primary": "ERROR", "description": str(e)[:200]}
        r["photo"] = p.name
        r["secondary"] = ",".join(r.get("secondary", []) or [])
        rows.append(r)
        print(p.name, r.get("primary"), r.get("severity"))

    out = pd.DataFrame(rows)
    out["model"] = a.model
    out["label_source"] = "inferred"
    loc = PROCESSED / "photo_locations.csv"
    if loc.exists():
        out = pd.read_csv(loc).merge(out, on="photo", how="right")
    out.to_csv(PROCESSED / "photo_labels_api.csv", index=False)
    print(out.groupby(["primary", "severity"]).size().to_string())


if __name__ == "__main__":
    main()
