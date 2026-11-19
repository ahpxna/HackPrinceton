# W34 perception: review of the first GPU run (Claude, 2026-11-18)

## The run
- Archive: `hf-perception-v1.tar.gz`, sha256 `76bcc06d…e6f5`. The sha was checked, and the archive is unpacked in `data/w34/hf-perception-v1/`.
- The GPU machine (RTX 4090, torch 2.10.0+cu128) ran both stages with exit 0:
  - SAM 3 used the `image` backend, not the fallback, and took 444 s for 486 faces;
  - OWLv2 took 243 s.
- Contact sheets and counts: `review_hf/` (`review.py`).
  - Each sheet shows the strongest instances and the 4 weakest ones still kept.
  - The SAM 3 masks line up exactly with the Mac's 1024-px faces, so the face indexing is the same on both machines.

## SAM 3: good enough to use (checked by eye on the sheets, no ground truth)

| Prompt | Instances / captures | Verdict |
|---|---|---|
| door | 757 / 78 | Correct, including open doors and close-ups (area up to 0.996). Weak ones at 0.40 are distant or ambiguous. |
| glass door | 56 / 35 | Correct: the patio slider (seen from many captures), the back door from outside, the front entry. |
| mirror | 27 / 11 | Correct: bathroom vanity mirrors. One miss at 0.45 is a microwave door. |
| staircase | 64 / 23 | Correct, and the mask also covers the balustrade. |
| railing | 124 / 41 | Not reviewed by eye. |
| shadow | 384 / 51 | **Mixed** (see below). |
| window | 864 / 70 | Not reviewed by eye. |
| person | 1 | Not reviewed. One instance of area 0.002. |
| tripod | 1 | Not reviewed. The staircase sheet shows a stand in a hallway (`kbstz8` f1). |
| television | 0 | Consistent: the house is vacant. |

The shadow masks hold three different things:
- **(a) The camera rig's own shadow** outdoors: a thin V on the ground that runs into the nadir patch on face 5. This is the transient shadow the atlas needs.
- **(b) Static cast shadows** of fences and trees. They are real, but they move during the capture session.
- **(c) Indoor false positives** at low scores: a garage floor, a wall patch, a door edge.

So only (a) should be masked. A candidate rule is face 5, score ≥ 0.7 and touching the nadir patch; it is not implemented yet.

**Tripod and nadir.** Matterport already blurs the nadir, so SAM 3 sees no tripod there. The nadir mask stays geometric.

## OWLv2: not reliable alone

**Counts are sightings, not objects.** For example, about 3 real bathtubs appear in 181 detections at ≥ 0.3 across 51 captures.

**Real, top-scoring:** bathtub (all top 16 correct).

**False:**
- **television:** 16 detections at ≥ 0.3, all wrong (light switches, mirror frames, wall plates). SAM 3 found 0 TVs, and the house is empty.
- **solar panel:** the top 16 (scores 0.36–0.52) are all wrong: ceiling HVAC registers, a floor grate, the gazebo roof, fences, neighbours' roofs. No solar panel is visible on this home in any of them.

**Mixed:** thermostat. Some are real wall thermostats; others are smoke detectors on the ceiling or an alarm keypad.

**Before any OWLv2 output can support a listing claim, it needs:**
1. A 3D raycast to the mesh, plus clustering across captures, so that one object counts once.
2. A `scope` label: `home` or `seen_from_home` (neighbours, the street).
3. A cross-check with SAM 3 for the same class where one exists (television, door, window).
4. Thresholds per class, set from reviewed examples rather than the global 0.10.

## Leads, not verified
- Portal `portal_0` (nav, band 0) uses capture `01rk816x…` in its pairs. That capture's face 4 has a SAM 3 door at 0.75 at the frame edge. Projecting the door masks onto the portal planes should confirm the portal and measure its width.
- The 11 mirror captures are candidates for the "room behind the mirror" artefacts in the PGSR or atlas outputs. Excluding mirror pixels from training is a cheap test.
