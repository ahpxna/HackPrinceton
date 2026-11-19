# W34 navigation: walkable floor grid and routes on the HF mesh

Code: `navgrid.py` (Claude, 2026-11-18). Outputs: `out_hf/`. Home: `HFwcH424n9w` (81 captures).

The design is ported from `robot-as-code-mac-mvp`:
- `src/robotcode/mapping.py` (`PriorMap`, distance grid);
- `src/robotcode/navigation.py` (`Costmap.cell_costs`: lethal radius plus exponential inflation cost; `plan_path`: 8-connected Dijkstra plus shortcutting).

That code plans on a single flat 2D slice. A house with stairs needs more, so the port changes the following.

| robot-as-code | navgrid.py |
|---|---|
| One 2D occupancy slice | Per-storey floor heights (2.5D) from a 5 cm voxelisation of the mesh. A floor cell is the top of an up-facing run with 0.25–1.90 m of headroom above it. |
| Obstacles inflated by a radius | Clearance = EDT × res, one radius per profile. Cliffs (\|Δh\| > step) count as walls. |
| 8-neighbour Dijkstra | The same, without corner cutting, plus stair links between the overlapping storey bands. |
| Exponential inflation cost | Centre-weighted cost `len·(1+2·exp(−(clear−r)/0.15))`, so routes keep to the middle of corridors. |
| Shortcutting | Binary-search line-of-sight shortcut that never crosses clearance < r or a step > max. |
| — | Profiles: person (r 0.20, step 0.25), wheelchair (r 0.38, step 0.05, turning circle 0.76). |
| — | Bottleneck width per capture pair (threshold sweep 0.10–1.50 m). |
| — | "Portals" from the Matterport sweep graph (see below). |

## Commands
```
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
$PY experiments/w34_navigation/navgrid.py build \
  --mesh experiments/w28_scene_reconstruction/matterport_export/assets \
  --captures data/splat/web/HFwcH424n9w/captures.json \
  --neighbours experiments/w32_world_assets/preview/build/world.json --out experiments/w34_navigation/out_hf
$PY experiments/w34_navigation/navgrid.py route --nav experiments/w34_navigation/out_hf \
  --captures data/splat/web/HFwcH424n9w/captures.json --from da1kcx --to bbpiyt [--profile wheelchair] [--portals]
```
A Mac build takes about 10 s (voxelise 2.1 s, layers + graph 0.8 s, analysis 6.8 s).

## Results (measured on the mesh; nothing checked by walking the house)

**Grid**
- Two storeys, floor z = 0.02 m and 3.07 m.
- 386,055 nodes and 1,444,551 edges.
- Free floor: 769.3 m² in band 0 and 195.9 m² in band 1. These totals include yards and other outdoor areas.

**Reachability by profile**

| Profile | Captures in the main component | Notes |
|---|---|---|
| Person (mesh only) | 60/81 | Main area: 62.4 m² on storey 0, 73.8 m² on storey 1. |
| Person + unverified portals | 76/81 | The 5 east-yard captures stay disconnected. |
| Wheelchair | 19/81 | 43.6 m² on storey 0, of which 22 m² allows a full turn. |

**Why 41 captures are blocked for a wheelchair**

| Cause | Captures |
|---|---|
| Other storey, reachable by stairs only | 37 |
| Passage narrower than the chair | 3 |
| No chair-wide floor within 0.6 m | 1 |

**Example routes**
- Person `da1kcx` → `bbpiyt`: 51.63 m over storeys 0 → 1, narrowest clearance 0.224 m, crossing `portal_0`.
- Wheelchair `t7yxz6` → `w1ybqw`: 13.8 m, narrowest clearance 0.391 m.

## Portals: assumed, not verified
- A portal is a pair of Matterport sweep neighbours whose straight line crosses exactly one thin blocked run (≤ 0.20 m) with floor on both sides.
- It is most likely a door that was shut during the scan, a glass door or a glass wall.
- The build found 9 portals. Each is marked `unverified` with `width_m: null`.
- Only the person profile with `--portals` uses them, and every route that does names the portals it crossed.
- A wheelchair may use a portal only once its width has been measured as ≥ 2r. None has been measured yet.
- The SAM 3 `door` / `glass door` masks from the W34 perception job can verify them: project the masks onto the portal plane, read off the opening width, and then set `width_m` and `verified`.

## Known limitations
- **Stair width is under-reported** at about 0.5 m, because the bands cut off part of the stair. The 37 "stairs only" wheelchair blocks are still correct.
- **Doors shut in the scan** are solid in the mesh. Without portals, rooms behind them show as unreachable, which explains 60/81 versus 76/81.
- **Outdoor areas:**
  - yards are counted as walkable floor;
  - tree-canopy noise appears in band 1;
  - the "main area" figures count only the component that holds the most captures.
- **No furniture model:** furniture counts as an obstacle only where the mesh has it.
- **Not wired in yet:**
  - the agent tool;
  - the route overlay in the viewer.
