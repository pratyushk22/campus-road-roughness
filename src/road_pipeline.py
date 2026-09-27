"""
Campus road-condition pipeline (phyphox exports: Accelerometer.csv, Location.csv, time.csv).

Usage:
    pip install pandas numpy pyarrow matplotlib
    python road_pipeline.py <ride_folder> <ride_id> "<seg names for each pause-chunk, comma-separated>"
e.g.
    python road_pipeline.py rp1/ rp1_1647 "R1,R2,R3,R4+R5"

What it does
  1. Vertical vibration: estimates gravity direction (1 s rolling mean), projects the
     dynamic acceleration onto it -> orientation-independent vertical vibration.
  2. Aligns GPS to accelerometer time, drops poor fixes (horizontal accuracy > 15 m),
     computes distance travelled.
  3. Splits into 50 m bins; per bin: RMS & peak vertical vibration, mean speed,
     speed-normalised roughness = RMS / max(speed, 1.5 m/s). Bins below 1.5 m/s are
     flagged (stopped/walking) and excluded from rankings.
  4. Detects bump events (|a_vert| > max(4 m/s^2, 6*MAD)), merges events < 2 s apart,
     geotags them -> photo checklist for the daylight pass.
  5. Segment labels come from phyphox PAUSE events (you paused at each landmark).

Outputs (in out/<ride_id>/):
    bins.parquet, events.parquet, events.csv (photo checklist), segments.csv, map.png
All rows carry source="observed".
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_M = 50
MIN_SPEED = 1.5      # m/s
MAX_HACC = 15        # m
EDGE_S = 15          # s trimmed after each START / before each PAUSE (sync taps, landmark stops)
EVENT_MIN_SPEED = 2.0
HOTSPOT_M = 25       # events closer than this are merged into one hotspot
FS_WIN = 100         # ~1 s at 100 Hz


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load(folder):
    f = Path(folder)
    acc = pd.read_csv(f / "Accelerometer.csv")
    acc.columns = ["t", "ax", "ay", "az"]
    loc = pd.read_csv(f / "Location.csv")
    loc.columns = ["t", "lat", "lon", "h", "v", "dir", "hacc", "vacc"]
    tm = pd.read_csv(f / "time.csv")
    return acc, loc, tm


def vertical_vibration(acc):
    a = acc[["ax", "ay", "az"]].to_numpy()
    g = pd.DataFrame(a).rolling(FS_WIN, center=True, min_periods=1).mean().to_numpy()
    gu = g / np.linalg.norm(g, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a - g, gu)


def main(folder, ride_id, seg_names):
    out = Path("out") / ride_id
    out.mkdir(parents=True, exist_ok=True)
    acc, loc, tm = load(folder)

    # ---- time anchors ----
    start_sys = tm.loc[tm.event == "START", "system time"].iloc[0]
    pauses = tm.loc[tm.event == "PAUSE", "experiment time"].to_numpy()
    # drop blip pauses (<2 s apart) and the final pause (end of ride)
    chunk_edges = [0.0]
    for p in pauses[:-1]:
        if p - chunk_edges[-1] > 2:
            chunk_edges.append(p)
    chunk_edges.append(np.inf)
    names = [s.strip() for s in seg_names.split(",")] if seg_names else []
    n_chunks = len(chunk_edges) - 1
    if len(names) != n_chunks:
        print(f"NOTE: {n_chunks} pause-chunks found, {len(names)} names given; "
              f"using C1..C{n_chunks}")
        names = [f"C{i+1}" for i in range(n_chunks)]

    # ---- GPS ----
    loc = loc[loc.hacc <= MAX_HACC].reset_index(drop=True)
    step = haversine(loc.lat.shift(), loc.lon.shift(), loc.lat, loc.lon).fillna(0)
    loc["dist"] = step.cumsum()

    # ---- vibration aligned to GPS ----
    acc["a_vert"] = vertical_vibration(acc)
    for col in ["lat", "lon", "v", "dist"]:
        acc[col] = np.interp(acc.t, loc.t, loc[col])
    acc = acc[(acc.t >= loc.t.min()) & (acc.t <= loc.t.max())].copy()
    tm2 = tm.copy()
    keep = np.ones(len(acc), bool)
    for e, et in zip(tm2.event, tm2["experiment time"]):
        if e == "START":
            keep &= ~((acc.t >= et) & (acc.t < et + EDGE_S))
        else:
            keep &= ~((acc.t > et - EDGE_S) & (acc.t <= et))
    acc = acc[keep].copy()
    acc["segment"] = pd.cut(acc.t, chunk_edges, labels=names, right=False)
    acc["bin"] = (acc.dist // BIN_M).astype(int)

    # ---- 50 m bins ----
    b = acc.groupby("bin").agg(
        segment=("segment", lambda s: s.mode().iloc[0]),
        t_start=("t", "min"), t_end=("t", "max"),
        lat=("lat", "mean"), lon=("lon", "mean"),
        speed=("v", "mean"),
        rms_vert=("a_vert", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        peak_vert=("a_vert", lambda x: float(np.max(np.abs(x)))),
        n=("a_vert", "size"),
    ).reset_index()
    b["valid"] = b.speed >= MIN_SPEED
    b["roughness"] = b.rms_vert / b.speed.clip(lower=MIN_SPEED)
    b["ride_id"] = ride_id
    b["timestamp"] = pd.to_datetime(start_sys + b.t_start, unit="s", utc=True) \
        .dt.tz_convert("Asia/Kolkata")
    b["source"] = "observed"
    b.to_parquet(out / "bins.parquet", index=False)

    # ---- bump events ----
    x = acc.a_vert.to_numpy()
    mad = np.median(np.abs(x - np.median(x)))
    thr = max(10.0, 10 * 1.4826 * mad)
    idx = np.where((np.abs(x) > thr) & (acc.v.to_numpy() >= EVENT_MIN_SPEED))[0]
    ev = []
    for i in idx:
        ti = acc.t.iloc[i]
        if ev and ti - ev[-1]["t"] < 2:
            if abs(x[i]) > ev[-1]["peak"]:
                ev[-1].update(t=ti, peak=abs(x[i]), lat=acc.lat.iloc[i],
                              lon=acc.lon.iloc[i], speed=acc.v.iloc[i],
                              segment=acc.segment.iloc[i])
            continue
        ev.append(dict(t=ti, peak=abs(x[i]), lat=acc.lat.iloc[i], lon=acc.lon.iloc[i],
                       speed=acc.v.iloc[i], segment=acc.segment.iloc[i]))
    ev = pd.DataFrame(ev)
    if len(ev):
        ev["peak_per_speed"] = ev.peak / ev.speed.clip(lower=MIN_SPEED)
        ev["time"] = pd.to_datetime(start_sys + ev.t, unit="s", utc=True) \
            .dt.tz_convert("Asia/Kolkata")
        ev["maps_link"] = [f"https://maps.google.com/?q={a:.6f},{o:.6f}"
                           for a, o in zip(ev.lat, ev.lon)]
        ev["ride_id"] = ride_id
        ev["source"] = "observed"
        ev = ev.sort_values("peak_per_speed", ascending=False).reset_index(drop=True)
        ev.insert(0, "event_id", [f"{ride_id}_E{i+1:02d}" for i in range(len(ev))])
        ev.to_parquet(out / "events.parquet", index=False)
        # hotspots: greedy merge of events within HOTSPOT_M (strongest first)
        hs = []
        for _, r in ev.iterrows():
            for h in hs:
                if haversine(h["lat"], h["lon"], r.lat, r.lon) < HOTSPOT_M:
                    h["n_events"] += 1
                    h["event_ids"] += "," + r.event_id
                    break
            else:
                hs.append(dict(lat=r.lat, lon=r.lon, segment=r.segment,
                               max_peak_per_speed=r.peak_per_speed, n_events=1,
                               event_ids=r.event_id))
        hs = pd.DataFrame(hs)
        hs["maps_link"] = [f"https://maps.google.com/?q={a:.6f},{o:.6f}"
                           for a, o in zip(hs.lat, hs.lon)]
        hs.insert(0, "hotspot_id", [f"{ride_id}_H{i+1:02d}" for i in range(len(hs))])
        hs["photo_taken"] = ""
        hs["defect_type"] = ""
        hs.to_csv(out / "photo_checklist.csv", index=False, float_format="%.6f")

    # ---- segment summary ----
    v = b[b.valid]
    seg = v.groupby("segment", observed=True).agg(
        length_m=("bin", lambda s: len(s) * BIN_M),
        mean_speed=("speed", "mean"),
        mean_roughness=("roughness", "mean"),
        p90_roughness=("roughness", lambda s: s.quantile(0.9)),
    )
    if len(ev):
        seg["hotspots"] = hs.groupby("segment", observed=True).size()
        seg["bump_events"] = ev.groupby("segment", observed=True).size()
        seg["bumps_per_km"] = seg.bump_events / (seg.length_m / 1000)
    seg = seg.fillna(0).round(3).sort_values("mean_roughness", ascending=False)
    seg.to_csv(out / "segments.csv")

    # ---- map ----
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(v.lon, v.lat, c=v.roughness, cmap="RdYlGn_r", s=45,
                    vmin=v.roughness.quantile(0.05), vmax=v.roughness.quantile(0.95))
    ax.scatter(b[~b.valid].lon, b[~b.valid].lat, c="lightgrey", s=20, label="slow/stopped")
    if len(ev):
        ax.scatter(hs.lon, hs.lat, marker="x", c="k", s=60, label="bump hotspot")
        for _, h in hs.head(10).iterrows():
            ax.annotate(h.hotspot_id.split("_")[-1], (h.lon, h.lat), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    for s, g in v.groupby("segment", observed=True):
        m = g.iloc[len(g) // 2]
        ax.annotate(str(s), (m.lon, m.lat), fontsize=11, weight="bold")
    plt.colorbar(sc, label="roughness (RMS vertical accel / speed)")
    ax.set_aspect(1 / np.cos(np.radians(v.lat.mean())))
    ax.set_title(f"Road roughness — {ride_id}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=130)

    print(f"Ride {ride_id}: {len(acc)} accel samples, {len(loc)} GPS fixes, "
          f"{b.dist.max() if 'dist' in b else acc.dist.max():.0f} m" if False else
          f"Ride {ride_id}: {len(acc)} accel samples, {len(loc)} GPS fixes, "
          f"{acc.dist.max():.0f} m, {len(b)} bins ({(~b.valid).sum()} slow), "
          f"{len(ev)} bump events in {len(hs) if len(ev) else 0} hotspots "
          f"(threshold {thr:.2f} m/s^2)")
    print(seg.to_string())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
