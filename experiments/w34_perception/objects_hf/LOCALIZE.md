# W34: OWLv2 detections placed in 3D (HF home), `w34-localize-7`

## Who did what
- **Claude wrote both scripts:**
  - `../localize.py`: the core;
  - `../localize_report.py`: sheets, maps and acceptance checks.
- **Codex CLI was briefed for the report part** (`experiments/briefs/w34_localize_report.md`). It hit its usage limit after reading the brief and wrote nothing. **Claude finished it.**
- **The checks below are therefore a labelled Claude self-check, not an independent Codex audit.**

## Commands (Mac CPU, `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`)

| Step | Command | Runtime |
|---|---|---|
| Localize | `python3.12 experiments/w34_perception/localize.py` | 148–177 s |
| Report | `python3.12 experiments/w34_perception/localize_report.py` | 207 s, of which the door check takes 189 s |

Outputs:
- `objects.json`: the objects, plus metadata (inputs and their sha256, method, `CLASS_RULES` with review notes, geometry check);
- `detections.jsonl`;
- `summary.json`;
- `acceptance.json`;
- `sheets/<label>.jpg`;
- `map_storey0.png` and `map_storey1.png`.

## Method (what is implemented)

**Raycasting.**
- Rays are cast exactly against the triangles of the W28 Matterport mesh (258,914 vertices, 494,569 triangles) with numpy Möller–Trumbore, culled per origin by bounding spheres. There are no new dependencies.
- Each detection casts 25 rays over the central 60 % of its box, plus the centre ray. Depth is the lower quartile of the hits.
- The point uses the W33 face convention (`center + (2x−1)·right + (2y−1)·down`). The down face is found with `argmin(basis[:,0,2])`.

**Per detection:**
- **Size:** angular size × depth.
- **Storey and `h_m`:** from the nav floors 0.02 m and 3.07 m.
- **`covered`:** an up-ray from 0.1 m in front of the point hits a surface within 8 m. This up-ray still uses the 5 cm voxel march.
- **`in_mirror`:** SAM 3 mirror bit 3. These detections are kept out of clustering.
- **`sam3_agrees`:** checked only for television ↔ television and garage door ↔ door.

**Scope:**
- no surface hit, or more than 5 m outside the convex hull of the capture positions → `seen_from_home`;
- within 2.5 m of the hull → `home_indoor` / `home_outdoor`, chosen by `covered`;
- anything in between → `unknown`.

**Clustering.** Leader clustering per label, strongest detection first.
- Radius = `clamp(0.5 × size_max, 0.25, 1.2)` m, with vertical distances halved.
- `size_m` is the 80th percentile of the members' sizes.

**Status:**
- `confirmed` needs all of:
  - support ≥ 2 captures with a score ≥ 0.3;
  - `score_max` ≥ the class threshold;
  - plausible under the `CLASS_RULES` priors;
  - scope ≠ `unknown`;
  - the class is `confirmable`;
  - no SAM 3 contradiction.
- Soft flags (support, unknown scope, unconfirmable class, weak SAM 3 agreement) make an object a `candidate`.
- Any other flag makes it `rejected`.
- **The class thresholds and `confirmable=False` were set by looking at this same home's views**, so they are a calibration, not a validated setting. Each rule's `note` records what was seen.

## Acceptance (real run)

**(a) Geometry against the exact mesh.**
- **Floor:** the centre ray of the down face hit the floor on 81/81 captures.
  - |hit − nav camera height|: median 0.028 m, p90 0.047 m, max 0.234 m. 75/81 are within 5 cm.
  - The worst three (`txqgap`, `y92xii`, `kbstz8`) have camera heights of 0.94–1.01 m (the rest have a median of 1.36 m). Their exact down-hits are at 1.18–1.21 m, and SAM 3 staircase masks cover 21–39 % of their down faces (median elsewhere: 0).
  - This **supports** the explanation "on stairs, and the nav floor snapped to a different step". It is not proven.
- **Depth, exact vs voxel** (300 seeded non-mirror detections): voxel − exact has a median of −0.042 m, so the voxel march always hits early. |diff| p90 is 0.216 m, the max is 10.2 m (rays through doorways), and 237/300 are within 10 cm.
  - The report re-casts the same rays and matches localize within 0.5 mm.

**(b) Door seen from two views.**
- Method: the centroid of each SAM 3 door mask (bit 6) is raycast to the mesh. A door in one capture is matched to a door in another by projecting the 3D point into the other capture's door box, with a line-of-sight test. Matching is never done by 3D proximity.
- 593 instances, 2,121 pairs:
  - all pairs: median 0.139 m, p90 0.562 m; 1,482 pairs (70 %) within 0.3 m;
  - masks not clipped by the frame edge: median 0.047 m, p90 0.332 m; 784/888 within 0.3 m.
- Example: `01rk816x…:4` ↔ `a84zix…:4` are 0.047 m apart, at (−0.64, −5.16, 2.10).

**(c) Reprojection of confirmed centroids into their member views.**
- 399/655 (60.9 %) land inside the box, and 455/655 (69.5 %) inside the box grown by 25 %.
- Split by view strength:
  - views at or above the class threshold: 179/199 (89.9 %);
  - weaker views: 276/451 (61.2 %).
- The cause: leader clusters also absorb weak views of nearby or partial things. The member lists are noisy; the strong views agree.

**(d) Expected outcomes. All pass.**

| Expectation | Result |
|---|---|
| Television: 0 confirmed | 0 |
| Solar panel: 0 confirmed with `scope=home*` | 0 |
| Bathtub: 2–4 confirmed, indoor, near the floor | 3 (`bathtub_113`, `_115`, `_116`), all upstairs, `h_m` 0.29–0.44, covered |
| The home's garage doors confirmed | `garage_door_428` (7.61, 5.59) and `_429` (7.42, 3.56). SAM 3 door agreement 0.67 / 0.75. Scope is **`home_indoor`**, which is misleading: the eave or the garage ceiling is above them (`above_m` 1.43 / 1.35). |
| Thermostat on a wall, or rejected | 1 confirmed: `thermostat_1566`, h 1.49 m, a real wall unit (checked by eye) |
| Smoke detector on the ceiling, or rejected | 22 confirmed. `above_m` median 0.09 m, max 0.33 m. 10 pairs are closer than 1 m, so duplicates are likely. |

**Eye review** (Claude, best views only, no ground truth):
- Reviewed in earlier versions: bathtub, garage door, thermostat, the top 8 smoke detectors, sink, shower, and an overview of the other confirmed classes.
- Reviewed for v7:
  - the 9 newly confirmed objects:
    - microwave_617, stove_1491, thermostat_1566 and toilets 1905–1907 are correct;
    - kitchen_island_581 and _583 are the same island, split into two clusters 1.55 m apart;
    - kitchen_island_582 is the perimeter counter beside the stove, not an island;
  - `ceiling_fan_241` at (−24.3, 8.0): a real fan/light under the backyard gazebo. The property is right, but the `home_indoor` scope is misleading.
- The storey-0 and storey-1 maps: capture dots sit on free floor, and confirmed objects lie inside the house, at the garage wall, or in the gazebo.

## Per-label summary (`summary.json`)

| label | det ≥0.1 | det ≥0.3 | in mirror | no surface | objects | confirmed | candidate | rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| air conditioner | 390 | 75 | 3 | 84 | 112 | 0 | 18 | 94 |
| bathtub | 801 | 181 | 20 | 0 | 96 | **3** | 20 | 73 |
| bed | 142 | 19 | 0 | 0 | 30 | 0 | 6 | 24 |
| ceiling fan | 162 | 80 | 2 | 6 | 40 | **7** | 12 | 21 |
| chandelier | 246 | 69 | 6 | 0 | 62 | **5** | 20 | 37 |
| clothes dryer | 5 | 1 | 2 | 0 | 3 | 0 | 0 | 3 |
| dishwasher | 173 | 27 | 8 | 0 | 32 | 0 | 19 | 13 |
| fireplace | 137 | 24 | 3 | 1 | 49 | **1** | 10 | 38 |
| garage door | 890 | 116 | 0 | 241 | 131 | **2** | 21 | 108 |
| hot tub | 76 | 0 | 0 | 1 | 26 | 0 | 0 | 26 |
| kitchen island | 263 | 58 | 5 | 0 | 36 | **3** | 15 | 18 |
| microwave | 71 | 31 | 4 | 0 | 17 | **1** | 6 | 10 |
| oven | 180 | 29 | 4 | 0 | 34 | 0 | 21 | 13 |
| range hood | 221 | 26 | 10 | 0 | 87 | 0 | 13 | 74 |
| refrigerator | 198 | 26 | 14 | 0 | 48 | 0 | 20 | 28 |
| shower | 1147 | 61 | 104 | 0 | 99 | 0 | 40 | 59 |
| sink | 358 | 75 | 35 | 2 | 58 | 0 | 14 | 44 |
| smoke detector | 1042 | 182 | 22 | 38 | 365 | **22** | 37 | 306 |
| sofa | 22 | 5 | 0 | 0 | 9 | 0 | 1 | 8 |
| solar panel | 1003 | 53 | 0 | 320 | 157 | 0 | 2 | 155 |
| stove | 147 | 35 | 7 | 0 | 33 | **1** | 14 | 18 |
| swimming pool | 18 | 0 | 0 | 0 | 14 | 0 | 0 | 14 |
| television | 53 | 16 | 6 | 0 | 28 | 0 | 0 | 28 |
| thermostat | 1127 | 301 | 45 | 17 | 339 | **1** | 36 | 302 |
| toilet | 315 | 48 | 19 | 5 | 143 | **3** | 13 | 127 |
| washing machine | 15 | 2 | 1 | 0 | 12 | 0 | 0 | 12 |
| water heater | 17 | 0 | 1 | 0 | 13 | 0 | 1 | 12 |
| wine cooler | 4 | 0 | 0 | 0 | 2 | 0 | 0 | 2 |

Totals: 2,075 objects and 49 confirmed, all `home_indoor`.

## Known failures
1. **`covered` is not the same as indoor.** Garage doors 428/429 and the gazebo fan 241 are labelled `home_indoor`. Read `home_indoor` as "home, under a roof".
2. **OWLv2 confusions are systematic, so multi-view agreement does not remove them.** For example, light switches score as thermostat, sink or TV from every view. They are handled only by class thresholds and `confirmable=False` (shower, sink, refrigerator, dishwasher, oven, range hood, air conditioner, bed), all tuned on this home.
3. **Duplicates and splits:**
   - smoke detectors: 10 pairs closer than 1 m among 22;
   - kitchen island: one island became 2 clusters, plus 1 wrong-class cluster.
   - Confirmed counts are therefore upper bounds, not object counts.
4. **Noisy member lists:** reprojection is only 61–70 % overall (90 % for strong views).
5. **The up-ray for `covered` still uses the voxel march,** which hits about 4 cm early with a long tail. This is harmless for an 8 m test but not exact.
6. **Thresholds are calibrated on the same home they are scored on.** There is no held-out home.
7. **`room_id` is not attached** (null).

## Not verified
- Candidates and most rejected objects were not reviewed by eye.
- The `seen_from_home` labels on neighbours' objects were not reviewed by eye in v7.
- The stair explanation for the 3 worst floor errors is supported but not proven.
- There are no ground-truth object counts; the only reference is Claude's eye review of best views.
