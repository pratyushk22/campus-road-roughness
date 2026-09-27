"""
Multi-ride aggregation for the campus road survey.

Every ride is map-matched onto a reference track (from rp4 plus rp2's direct road), so
segments line up by position, not by where the rider paused.

Segments (reference landmarks):
  S1 Faculty market - Serpentine        (user's R5)
  S2 Serpentine - New SAC               (R4)
  S3 New SAC - Library                  (R3)
  S4 Library - Core 5                   (R2)
  S5 Core 5 - Brahmaputra (backside)    (R1)
  S6 Library - Brahmaputra direct road  (R6, only ridden in rp2)

Usage:
    python src/multi_ride.py        # edit RIDES below to add ride folders
Reads data/raw/rides/<ride>/, writes to data/processed/:
    samples_binned.parquet   per ride x 50 m canonical bin
    segment_summary.csv      mean +- std roughness across rides
    hotspots.csv             bump hotspots, confirmed if seen in >= 2 rides
    combined_map.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from road_pipeline import haversine, vertical_vibration, MIN_SPEED

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "rides"
PROCESSED = BASE / "data" / "processed"

# folder, ride_id, direction ("out" = Brahmaputra->FM, "ret" = FM->Brahmaputra)
RIDES = [
    ("rp2", "rp2_0926_1647", "ret"),
    ("rp3", "rp3_0927_1146", "out"),
    ("rp4", "rp4_0927_1218", "ret"),
    ("rp5", "rp5_0927_1407", "out"),
    ("rp6", "rp6_0927_1433", "ret"),   # return leg with photo stops
]
# shorter trim around pauses for rides with photo stops (bump sits just before the stop)
EDGE_OVERRIDE = {"rp6_0927_1433": 4}
REF_RIDE = "rp4"
# rp4 pause-chunk -> segment (chunks: FM|4.1|4.2|NewSAC|Library|Core5|4.6|end)
REF_CHUNK_SEG = ["S1", "S2", "S2", "S3", "S4", "S5", "S5"]
S6_RIDE, S6_FROM_T = "rp2", 740.2          # rp2 after its Library pause = direct road
SEG_NAMES = {
    "S1": "Faculty market–Serpentine", "S2": "Serpentine–New SAC",
    "S3": "New SAC–Library", "S4": "Library–Core 5",
    "S5": "Core 5–Brahmaputra (backside)", "S6": "Library–Brahmaputra (direct)",
}
BIN_M, MATCH_M, EDGE_S, HOTSPOT_M = 50, 30, 15, 25
LAT0 = 26.19
M_PER_DEG = 111_320.0


def xy(lat, lon):
    return np.c_[(np.asarray(lon) - 91.69) * M_PER_DEG * np.cos(np.radians(LAT0)),
                 (np.asarray(lat) - LAT0) * M_PER_DEG]


def read(folder):
    f = Path(folder) if Path(folder).exists() else RAW / folder
    acc = pd.read_csv(f / "Accelerometer.csv"); acc.columns = ["t", "ax", "ay", "az"]
    loc = pd.read_csv(f / "Location.csv")
    loc.columns = ["t", "lat", "lon", "h", "v", "dir", "hacc", "vacc"]
    tfile = f / "time.csv" if (f / "time.csv").exists() else f / "meta" / "time.csv"
    tm = pd.read_csv(tfile)
    return acc, loc[loc.hacc <= 15].reset_index(drop=True), tm


def chunk_edges(tm):
    ps = tm.loc[tm.event == "PAUSE", "experiment time"].to_numpy()
    e = [0.0]
    for p in ps[:-1]:
        if p - e[-1] > 2:
            e.append(p)
    return e + [np.inf]


def build_reference():
    _, loc, tm = read(REF_RIDE)
    e = chunk_edges(tm)
    loc["seg"] = pd.cut(loc.t, e, labels=False, right=False).map(
        lambda i: REF_CHUNK_SEG[min(int(i), len(REF_CHUNK_SEG) - 1)])
    ref = [loc[["lat", "lon", "seg"]]]
    _, l2, _ = read(S6_RIDE)
    s6 = l2[l2.t > S6_FROM_T][["lat", "lon"]].assign(seg="S6")
    # keep only the part of rp2's tail that is NOT near the rp4 track
    tree = cKDTree(xy(loc.lat, loc.lon))
    d, _ = tree.query(xy(s6.lat, s6.lon))
    ref.append(s6[d > MATCH_M])
    ref = pd.concat(ref, ignore_index=True)
    # distance along each segment (per-segment cumulative)
    ref["d_seg"] = 0.0
    for s, g in ref.groupby("seg"):
        st = haversine(g.lat.shift(), g.lon.shift(), g.lat, g.lon).fillna(0)
        ref.loc[g.index, "d_seg"] = st.cumsum().to_numpy()
    return ref


def process_ride(folder, ride_id, direction, ref, tree):
    acc, loc, tm = read(folder)
    rate = 1 / acc.t.diff().median()
    if rate < 50:
        print(f"SKIP {ride_id}: accelerometer at {rate:.1f} Hz (need ~100 Hz). "
              f"GPS still used for the reference track.")
        return None, None
    acc["a_vert"] = vertical_vibration(acc)
    for c in ["lat", "lon", "v"]:
        acc[c] = np.interp(acc.t, loc.t, loc[c])
    keep = (acc.t >= loc.t.min()) & (acc.t <= loc.t.max())
    edge = EDGE_OVERRIDE.get(ride_id, EDGE_S)
    for ev, et in zip(tm.event, tm["experiment time"]):
        if ev == "START":
            keep &= ~((acc.t >= et) & (acc.t < et + edge))
        else:
            keep &= ~((acc.t > et - edge) & (acc.t <= et))
    acc = acc[keep].copy()
    d, i = tree.query(xy(acc.lat, acc.lon))
    acc = acc[d <= MATCH_M].copy()
    i = i[d <= MATCH_M]
    acc["seg"] = ref.seg.to_numpy()[i]
    acc["bin"] = (ref.d_seg.to_numpy()[i] // BIN_M).astype(int)

    b = acc.groupby(["seg", "bin"]).agg(
        lat=("lat", "mean"), lon=("lon", "mean"), speed=("v", "mean"),
        rms_vert=("a_vert", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        n=("a_vert", "size")).reset_index()
    b = b[(b.n >= 50) & (b.speed >= MIN_SPEED)]
    b["roughness"] = b.rms_vert / b.speed.clip(lower=MIN_SPEED)
    b["ride_id"], b["direction"], b["source"] = ride_id, direction, "observed"

    x = acc.a_vert.to_numpy()
    mad = np.median(np.abs(x - np.median(x)))
    thr = max(10.0, 10 * 1.4826 * mad)
    idx = np.where((np.abs(x) > thr) & (acc.v.to_numpy() >= 2.0))[0]
    evs = []
    for k in idx:
        r = acc.iloc[k]
        if evs and r.t - evs[-1]["t"] < 2:
            if abs(r.a_vert) > evs[-1]["peak"]:
                evs[-1].update(t=r.t, peak=abs(r.a_vert), lat=r.lat, lon=r.lon, speed=r.v)
            continue
        evs.append(dict(t=r.t, peak=abs(r.a_vert), lat=r.lat, lon=r.lon, speed=r.v,
                        seg=r.seg))
    ev = pd.DataFrame(evs)
    ev["peak_per_speed"] = ev.peak / ev.speed.clip(lower=MIN_SPEED)
    ev["ride_id"] = ride_id
    return b, ev


def main():
    out = PROCESSED; out.mkdir(parents=True, exist_ok=True)
    ref = build_reference()
    tree = cKDTree(xy(ref.lat, ref.lon))
    bins, evs = [], []
    for folder, rid, direction in RIDES:
        b, e = process_ride(folder, rid, direction, ref, tree)
        if b is None:
            continue
        bins.append(b); evs.append(e)
        print(f"{rid}: {len(b)} bins, {len(e)} bump events")
    bins = pd.concat(bins, ignore_index=True)
    evs = pd.concat(evs, ignore_index=True)
    bins.to_parquet(out / "samples_binned.parquet", index=False)

    # ---- segment summary: per-ride means, then mean +- std across rides ----
    # a ride counts for a segment only if it covered >= 50% of that segment's bins
    # (e.g. rp2 only brushes the start of S4 at the Library junction)
    nb = bins.groupby("seg").bin.nunique()
    cov = bins.groupby(["seg", "ride_id"]).bin.nunique().unstack() .div(nb, axis=0)
    ok = cov.stack()[lambda x: x >= 0.5].index
    full = bins.set_index(["seg", "ride_id"]).loc[lambda d: d.index.isin(ok)].reset_index()
    per_ride = full.groupby(["seg", "ride_id"]).roughness.mean().unstack()
    seg = pd.DataFrame({
        "segment": [SEG_NAMES[s] for s in per_ride.index],
        "rides": per_ride.notna().sum(axis=1),
        "mean_roughness": per_ride.mean(axis=1),
        "std_across_rides": per_ride.std(axis=1),
    }, index=per_ride.index).join(per_ride.add_prefix("r_"))
    length = bins.groupby("seg").bin.nunique() * BIN_M
    seg["length_m"] = length

    # ---- hotspots across rides ----
    evs = evs.sort_values("peak_per_speed", ascending=False).reset_index(drop=True)
    hs = []
    for _, r in evs.iterrows():
        for h in hs:
            if haversine(h["lat"], h["lon"], r.lat, r.lon) < HOTSPOT_M:
                h["rides"].add(r.ride_id); h["n_events"] += 1
                break
        else:
            hs.append(dict(lat=r.lat, lon=r.lon, seg=r.seg, rides={r.ride_id},
                           n_events=1, max_peak_per_speed=r.peak_per_speed))
    hs = pd.DataFrame(hs)
    rides_on_seg = seg["rides"].reindex(list(SEG_NAMES)).fillna(0)
    hs["n_rides_seen"] = hs.rides.map(len)
    hs["rides_covering_segment"] = hs.seg.map(rides_on_seg)
    hs["status"] = np.where(hs.n_rides_seen >= 2, "confirmed",
                            np.where(hs.rides_covering_segment >= 2, "single-ride",
                                     "unverified (1 pass)"))
    hs["segment"] = hs.seg.map(SEG_NAMES)
    hs["rides"] = hs.rides.map(lambda s: ",".join(sorted(s)))
    hs = hs.sort_values(["n_rides_seen", "max_peak_per_speed"], ascending=False) \
           .reset_index(drop=True)
    hs.insert(0, "hotspot_id", [f"H{i+1:02d}" for i in range(len(hs))])
    hs["maps_link"] = [f"https://maps.google.com/?q={a:.6f},{o:.6f}"
                       for a, o in zip(hs.lat, hs.lon)]
    hs["photo_file"], hs["defect_type"], hs["source"] = "", "", "observed"
    hs.to_csv(out / "hotspots.csv", index=False, float_format="%.6f")

    conf = hs[hs.status == "confirmed"].groupby("seg").size()
    seg["confirmed_hotspots"] = conf
    seg["confirmed_per_km"] = seg.confirmed_hotspots / (seg.length_m / 1000)
    seg = seg.fillna({"confirmed_hotspots": 0, "confirmed_per_km": 0}) \
             .sort_values("mean_roughness", ascending=False).round(3)
    seg.to_csv(out / "segment_summary.csv")

    # ---- map ----
    m = bins.groupby(["seg", "bin"]).agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                         r=("roughness", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(m.lon, m.lat, c=m.r, cmap="RdYlGn_r", s=55,
                    vmin=m.r.quantile(.05), vmax=m.r.quantile(.95))
    c = hs[hs.status == "confirmed"]
    ax.scatter(c.lon, c.lat, marker="x", c="k", s=70, label="confirmed hotspot (≥2 rides)")
    for _, h in c.head(12).iterrows():
        ax.annotate(h.hotspot_id, (h.lon, h.lat), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    for s, g in m.groupby("seg"):
        mid = g.iloc[len(g) // 2]
        ax.annotate(s, (mid.lon, mid.lat), fontsize=12, weight="bold", color="navy",
                    xytext=(-18, 6), textcoords="offset points")
    plt.colorbar(sc, label="mean roughness across rides (RMS vertical accel / speed)")
    ax.set_aspect(1 / np.cos(np.radians(LAT0)))
    ax.set_title("Campus road roughness — all rides combined")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(BASE / "outputs" / "combined_map.png", dpi=130)
    seg.to_parquet(out / "segment_summary.parquet")
    hs.to_parquet(out / "hotspots.parquet", index=False)

    print(seg.to_string())
    print(hs.status.value_counts().to_string())


if __name__ == "__main__":
    main()
