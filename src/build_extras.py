"""
Builds the remaining processed tables and the data sample.
    python src/build_extras.py
- data/processed/photo_labels.parquet  (defect labels from the vision model; source=inferred)
- data/sample/                         (small slices of every table)
"""
import pandas as pd
from multi_ride import BASE, PROCESSED, RAW

lab = pd.read_csv(PROCESSED / "photo_labels.csv")
lab.to_parquet(PROCESSED / "photo_labels.parquet", index=False)

smp = BASE / "data" / "sample"
smp.mkdir(exist_ok=True)
pd.read_csv(RAW / "rp5" / "Accelerometer.csv", nrows=2000).to_csv(smp / "rp5_accelerometer_first2000.csv", index=False)
pd.read_csv(RAW / "rp5" / "Location.csv", nrows=60).to_csv(smp / "rp5_location_first60.csv", index=False)
pd.read_parquet(PROCESSED / "samples_binned.parquet").head(30).to_csv(smp / "binned_head30.csv", index=False)
pd.read_csv(PROCESSED / "hotspots.csv").head(15).to_csv(smp / "hotspots_top15.csv", index=False)
lab.head(10).to_csv(smp / "photo_labels_head10.csv", index=False)
print("ok")
