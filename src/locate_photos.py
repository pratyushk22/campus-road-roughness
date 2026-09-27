"""
Place each photo on the map using the rp6 ride timeline (photos carry no GPS, only a
timestamp): the photo time is mapped to rp6 experiment time via time.csv (photos were
taken while phyphox was paused), then to rp6's GPS position, the nearest reference
segment, and the nearest bump hotspot.

    python src/locate_photos.py
Reads data/raw/photo_times.csv, data/raw/rides/rp6/, data/processed/hotspots.csv
Writes data/processed/photo_locations.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import multi_ride as m

PHOTO_RIDE = "rp6"


def main():
    t = pd.read_csv(m.RAW / PHOTO_RIDE / "time.csv")
    t["sys"] = pd.to_datetime(t["system time text"].str.replace(" UTC+05:30", "", regex=False))
    _, loc, _ = m.read(PHOTO_RIDE)
    ref = m.build_reference()
    tree = cKDTree(m.xy(ref.lat, ref.lon))
    hs = pd.read_csv(m.PROCESSED / "hotspots.csv")
    ph = pd.read_csv(m.BASE / "data" / "raw" / "photo_times.csv")
    ph["sys"] = pd.to_datetime(ph.time, format="%Y:%m:%d %H:%M:%S")
    rows = []
    for _, p in ph.iterrows():
        r = t[t.sys <= p.sys].iloc[-1]
        et = r["experiment time"] if r.event == "PAUSE" else \
            r["experiment time"] + (p.sys - r.sys).total_seconds()
        i = (loc.t - et).abs().argmin()
        la, lo = loc.lat.iloc[i], loc.lon.iloc[i]
        _, j = tree.query(m.xy([la], [lo]))
        dh = m.haversine(la, lo, hs.lat.values, hs.lon.values)
        k = int(np.argmin(dh))
        rows.append(dict(photo=p.photo, time=p.time, lat=round(la, 6), lon=round(lo, 6),
                         segment=m.SEG_NAMES[ref.seg.iloc[j[0]]],
                         nearest_hotspot=hs.hotspot_id[k], hotspot_dist_m=round(dh[k], 1),
                         hotspot_status=hs.status[k]))
    out = pd.DataFrame(rows)
    out.to_csv(m.PROCESSED / "photo_locations.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
