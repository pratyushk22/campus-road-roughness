"""
Summarise the anonymous road survey.
    python src/survey_summary.py
Reads data/raw/survey_responses.csv -> data/processed/survey.parquet, outputs/survey_summary.md
"""
import pandas as pd
from multi_ride import BASE, PROCESSED

s = pd.read_csv(BASE / "data" / "raw" / "survey_responses.csv")
s.columns = ["timestamp", "most_used_roads", "travel_mode", "when_used", "worst_road",
             "incident_fall_injury_or_splash", "other_comments"]
s["timestamp"] = pd.to_datetime(s.timestamp, dayfirst=True, format="mixed")
s["source"] = "self_reported"
s.to_parquet(PROCESSED / "survey.parquet", index=False)


def multi(col):
    return s[col].dropna().str.split(", ").explode().str.strip().value_counts()


n = len(s)
lines = [f"# Survey summary (n = {n}, anonymous)\n"]
for title, vc in [("Most-used roads (pick 3)", multi("most_used_roads")),
                  ("Travel mode", s.travel_mode.value_counts()),
                  ("When used", multi("when_used")),
                  ("Worst road", s.worst_road.value_counts()),
                  ("Fallen / injured / splashed", s.incident_fall_injury_or_splash.value_counts())]:
    lines.append(f"## {title}\n")
    lines += [f"- {k}: {v} of {n}" for k, v in vc.items()]
    lines.append("")
lines.append("## Free-text comments\n")
lines += [f"- {c.strip()}" for c in s.other_comments.dropna() if c.strip()]
out = BASE / "outputs" / "survey_summary.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
