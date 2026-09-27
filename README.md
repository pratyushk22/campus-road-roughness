# Campus Road Roughness — IIT Guwahati (LATENT48, SAIL × Granica)

**Question:** Which campus road stretches should be repaired first, measured rather than guessed?
**Decision-maker:** IITG Estates / civil maintenance. **Affected users:** students, delivery drivers, professors and their kins and everyone else who uses IITG roads.

I mounted a phone on a bicycle handlebar, rode a fixed ~5 km loop in both directions across
two days, and turned vibration + GPS into a per-50 m roughness map and a list of bump
hotspots. Photos at hotspots were classified by a vision model, and an anonymous student
survey was used as a cross-check.
---

## Headline findings

| Segment | Roughness (mean ± std across rides) | Rides | Photo evidence |
|---|---|---|---|
| **Library – Core 5** | **0.85 ± 0.03** | 3 | fatigue (alligator) cracking, broken slab joints, surface delamination |
| New SAC – Library | 0.54 ± 0.07 | 4 | — |
| Faculty market – Serpentine | 0.53 ± 0.06 | 4 | raveling, failing patches |
| Serpentine – New SAC | 0.51 ± 0.07 | 3 | paver–asphalt joint gap |
| Library – Brahmaputra (direct) | 0.50 | 1 | — |
| Core 5 – Brahmaputra (backside) | 0.47 ± 0.03 | 3 | broken / sunken / missing pavers at a few crossings, drainage |

Roughness = RMS vertical acceleration ÷ speed (m/s² per m/s), per 50 m bin, averaged per ride then across rides.

1. **Library – Core 5 is the worst road end-to-end** (~1.6× the others, consistent in both directions). Photos show structural failure → resurfacing, not patching.
2. **Library junction** holds the strongest bump in the dataset (H01), detected on **all 4 usable rides** across two days and both directions; photos show a slab-edge step and standing water.
3. **Backside (Core 5 – Brahmaputra)** is smooth *on average* but contains the most severe spot defects (broken paver strips across the road, sunken edges: H08, H15). Half the survey respondents (5 of 10) rate it worst — consistent with sharp local hazards and waterlogging, which an average hides.

Evidence strength: 88 bump hotspots, **55 confirmed** (seen in ≥ 2 rides). 23 photos at **15 distinct
spots**; all 15 lie within 25 m of a sensor hotspot, 11 of them confirmed ones. Caveat: the rider
chose photo stops partly from the hotspot list, so this shows flagged spots are real defects, not
that the sensor finds every defect.

---

## Repository layout

```
data/raw/rides/<ride>/     phyphox exports: Accelerometer.csv, Location.csv, time.csv, device.csv
data/raw/photos/           23 defect photos (JPEG, downsized; converted from HEIC)
data/raw/photo_times.csv   photo capture times (EXIF DateTimeOriginal; photos carried no GPS)
data/raw/survey_responses.csv  anonymous Google Form export
data/processed/            analysis tables (Parquet + CSV mirrors)
data/sample/               small slices of each table for quick inspection
src/road_pipeline.py       single-ride processing (vertical vibration, bins, bump events)
src/multi_ride.py          all rides: map-matching, segment stats, cross-ride hotspots, map
src/locate_photos.py       places photos via the rp6 timeline, matches to hotspots
src/classify_photos.py     vision-model step: photo -> defect type + severity (Anthropic API)
src/survey_summary.py      survey -> survey.parquet + outputs/survey_summary.md
src/build_extras.py        photo-label Parquet, data sample
src/run_all.py             runs everything + writes outputs/processing_report.md
outputs/                   processing_report.md, processing_steps.png, combined_map.png,
                           H08_photo_vs_spike.png, survey_summary.md
```

### Reproduce
```bash
pip install -r requirements.txt
python src/run_all.py         # whole pipeline + processing report (per-ride data flow)

# vision step (needs your own Anthropic API key)
pip install anthropic && export ANTHROPIC_API_KEY=...
python src/classify_photos.py # -> data/processed/photo_labels_api.csv
```
`outputs/processing_report.md` shows, for every ride, how many samples survived each stage
(raw → trimmed → map-matched → moving) and why rp4 was excluded; `outputs/processing_steps.png`
shows raw 3-axis signal → vertical vibration with the bump threshold → 50 m roughness.

---

## Collection

| Ride | Start (IST) | Direction | Duration | Accel rate | Accel samples | GPS fixes | Pauses | Status |
|---|---|---|---|---|---|---|---|---|
| rp2 | 26 Sep 16:47 | Faculty market → Brahmaputra (via Library–Brahmaputra direct road) | 17.4 min | 100 Hz | 104,788 | 1,046 | 5 | used |
| rp3 | 27 Sep 11:46 | Brahmaputra → Faculty market (via Core 5) | 23.1 min | 100 Hz | 138,857 | 1,385 | 6 | used |
| rp4 | 27 Sep 12:18 | Faculty market → Brahmaputra (via Core 5) | 22.6 min | **0.5 Hz** | 681 | 1,391 | 7 | GPS only (see below) |
| rp5 | 27 Sep 14:07 | Brahmaputra → Faculty market | 20.8 min | 100 Hz | 125,459 | 1,273 | 5 | used |
| rp6 | 27 Sep 14:33 | Faculty market → Brahmaputra, with photo stops | 18.6 min | 100 Hz | 111,675 | 1,177 | 21 | used |

**Protocol.** iPhone fixed on the handlebar in the same orientation every ride; phyphox logging
acceleration (~100 Hz) and GPS (~1 Hz). Same bicycle, steady ~12 km/h (mean GPS speed 3.2–3.4 m/s),
riding straight over the road surface. Before each ride the rider stood still and tapped the
handlebar three times (sync marker). phyphox was paused at each landmark (Faculty market,
Serpentine, New SAC, Library, Core 5, Brahmaputra) and, on rp6, at each photo stop: ride over the
defect at normal speed, stop ~10 m later, pause, photograph, resume.

**Survey.** Anonymous Google Form (no names, emails or roll numbers) shared in hostel groups:
most-used roads, travel mode, time of use, worst road, incidents (fall / injury / splashing), free
comments. **10 responses** at time of submission: 7 of 10 mostly cycle; **7 of 10 report a fall,
injury or waterlogging splash**; most-used roads Core 5 – Library (8 of 10) and Brahmaputra – Core 5
backside (7 of 10); rated worst: **Brahmaputra – Core 5 backside (5 of 10)**. Full breakdown in
`outputs/survey_summary.md`.

### What went wrong, and what we did
- **Sensor silently throttled (rp4).** The accelerometer dropped to 0.5 Hz, almost certainly
  because the screen auto-locked. An automatic sample-rate check in `multi_ride.py` excludes any
  ride below 50 Hz; rp4's GPS is still used as the reference route. Fixed for later rides by
  disabling auto-lock and keeping phyphox in the foreground; rp5/rp6 recorded at 100 Hz.
- **Routes varied between rides** (rp2 returned by a different road). Handled by map-matching every
  ride onto a single reference track instead of relying on pause order.
- **Speed and slope affect vibration.** Roughness is normalised by speed; bins under 1.5 m/s are
  excluded; 15 s around each start/pause is trimmed (sync taps, landmark stops) — 4 s on rp6 so a
  bump just before a photo stop still counts.
- **Directions see different lanes.** Hotspots are confirmed only when seen in ≥ 2 rides;
  segment scores are averaged per ride, then across rides.
- **Lost upload.** The first outbound ride (rp1) was overwritten by a same-named upload and is not
  included.
- **Photos had no GPS.** Located via their capture time on the rp6 pause timeline instead.

---

## Method

1. **Vertical vibration.** Gravity direction = 1 s rolling mean of the 3-axis signal; dynamic
   acceleration projected onto it → orientation-independent vertical vibration.
2. **Map-matching.** Reference track = rp4 GPS (segments from its landmark pauses) + rp2's
   direct Library–Brahmaputra road. Every sample is snapped to the nearest reference point
   (≤ 30 m) → segment and 50 m bin.
3. **Roughness per bin** = RMS(vertical) ÷ max(speed, 1.5 m/s). A ride counts for a segment only if
   it covers ≥ 50 % of that segment's bins.
4. **Bump events**: |vertical| > max(10 m/s², 10 × robust σ) at speed ≥ 2 m/s, merged within 2 s.
   **Hotspots**: events from all rides merged within 25 m; *confirmed* = seen in ≥ 2 rides.
5. **Photos → defects (AI).** Each photo was classified by a vision model (Claude, Anthropic)
   into defect type and severity (low / medium / high) with a one-line description. The shipped
   labels (`photo_labels.csv`) were produced through the Claude chat interface during the event;
   `src/classify_photos.py` reproduces the step via the API with a fixed category list and a
   JSON output format (prompt in the script). Labels are stored as `inferred` and should be
   spot-checked; none were edited to fit the sensor results.
6. **Survey** summarised as counts; used only as an independent cross-check.

---

## Schema (data/processed)

**samples_binned.parquet** — one row per ride × 50 m canonical bin
| column | type | meaning |
|---|---|---|
| seg | str | segment id S1–S6 |
| bin | int | 50 m bin index along the segment |
| lat, lon | float | mean position of the bin in this ride |
| speed | float | mean GPS speed, m/s |
| rms_vert | float | RMS vertical acceleration, m/s² |
| n | int | accelerometer samples in bin |
| roughness | float | rms_vert / max(speed, 1.5) |
| ride_id | str | e.g. rp5_0927_1407 |
| direction | str | out (Brahmaputra→Faculty market) / ret |
| source | str | `observed` |

**segment_summary.parquet** — one row per segment: `segment`, `rides`, `mean_roughness`,
`std_across_rides`, `r_<ride_id>` (per-ride mean), `length_m`, `confirmed_hotspots`, `confirmed_per_km`.

**hotspots.parquet** — one row per bump hotspot: `hotspot_id`, `lat`, `lon`, `seg`, `segment`,
`rides` (which rides detected it), `n_events`, `max_peak_per_speed`, `n_rides_seen`,
`rides_covering_segment`, `status` (confirmed / single-ride / unverified (1 pass)), `maps_link`,
`source`=`observed`.

**photo_labels.parquet** — one row per photo: `photo`, `time`, `lat`, `lon` (from rp6 GPS at capture
time), `segment`, `nearest_hotspot`, `hotspot_dist_m`, `hotspot_status`, `hotspot_rides`,
`defect_type`, `severity`, `description`, `label_source`=`inferred`, `location_source`=`observed`.

**survey.parquet** — one row per response: `timestamp`, `most_used_roads`, `travel_mode`, `when_used`,
`worst_road`, `incident_fall_injury_or_splash`, `other_comments`, `source`=`self_reported`.

**Provenance:** `observed` = measured by us; `inferred` = model output; `self_reported` = survey.
**No synthetic data** is used anywhere in this dataset.

---

## Limitations
- Weekend data, 2 days, one rider, one bicycle and phone; roughness is relative, not a calibrated
  road-roughness index (IRI).
- 3–4 usable passes per segment; Library – Brahmaputra direct has a single pass.
- Vibration measures ride quality; it does not see waterlogging, lighting or edge drops outside
  the wheel path — photos and survey fill part of that gap.
- Photo stops were chosen partly from the hotspot list (selection bias).
- Survey is a small convenience sample (10) from the rider's own groups.

## What we would collect next
Weekday rides and fixed-point traffic counts for usage weighting; a second phone/rider to
test device dependence; a rainy-day ride for waterlogging; more survey responses across hostels.

## Privacy
No people are identifiable in the data: counts and road photos only; one photo (IMG_2542) was
cropped to remove a distant cyclist. The survey collected no names, emails or roll numbers.
Raw full-resolution photos and video are not published.
