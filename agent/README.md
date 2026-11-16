# Sunproof buyer agent

Run from `/Users/phanan/HackPrinceton` using `.venv-agent/bin/python`. `requirements.txt` pins the installed environment.

## Personal sunlight fit

`get_sun_habits` asks about early rising, office room, naps, cooling priorities and
plants. `remember_sun_habits` requires a separate human approval and merges the
answers into `/memories/<profile>.json` (`agent/state/` on disk), preserving needs
and feedback. Missing answers remain unknown; profiles stay isolated.

`rank_sun_fit(needs)` filters hard needs, excludes unreliable placement, and ranks
the F4 cohort when available (the older six-home F3 fallback stays provisional).
Equal weights (+1 wanted, -1 avoided) apply to
morning kitchen sun, work-hour office exposure, nap-time bedroom exposure, summer
bedroom noon/afternoon exposure, cooling-demand exposure and winter living-room
plant placement, according to approved habits. Score is the mean of signed rule
exposure fractions, averaged equally over covered rooms/dates. Coverage is reported
and sorted before score. For the F3 fallback the metric is clear window aperture fraction weighted by
glass area, converted to equivalent minutes; it is not measured room-floor sun,
actual glare, thermal load or cooling cost. Local timezone/DST is applied to UTC
hour-ending bin midpoints. Geometry scenario is used consistently; DSM and weather
are not included in this provisional score. Only four representative dates exist.

Bare Bedroom labels do not identify a "second bedroom". The tool proposes the
non-primary bedroom with least mean work-hour window exposure as an office and scores the
remaining bedrooms for summer sleeping exposure. This is a proposed room-use
assignment, never an ordinal identification; ask whether it suits the buyer.
Explicit `work_room_ids` selections override suggestions. Default work/nap/plant
bands require buyer confirmation; no indoor temperature is stated.

If `data/facts/daylight/room_sun_f4.json` exists with schema_version 1 and rooms,
the tool reads its per-room heat, hot_sun_hours_jja, flags/bands and provenance.
Rooms marked not_ranked_no_sun are withheld. Missing heat/hot-sun fields say
"not computed yet"; no heat estimates are invented. F4 ranking uses actual local
hourly sun and flags, and summer components include clear-glass peak solar heat
scaled as W/(W+1500 W) and hot-sun hours divided by 24. Components, rooms and
rules have equal weights; missing criteria remain unscored. These are declared
preference heuristics, not calibrated comfort or cooling-cost estimates. Twelve
representative days and historical weather underpin F4. The tool compares all
eligible cohort homes and returns the top ten with coverage and source files.
Computed F4 evidence is not provisional; proposed office use still needs confirmation.

Run `.venv-agent/bin/python -m agent.eval --dry-run`. The sunlight case records
actual Deep Agents tool calls, memory approval and persistence, rankings, room
reasons and evidence files in `data/claims/eval/sun_fit_dry_run_trace.json`.
Natural-language decisions are scripted: this proves the harness, not a live model.
Evaluation workspaces remain under `data/claims/eval/workspaces/`.

Changes in this round were implemented by Codex CLI. To inspect reversible edit
history use `python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path <file>`,
then `rollback <id>` (dry run), or `rollback <id> --yes` to apply.

## Local panorama features

Optional dependencies are pinned in `requirements-panorama.txt`. The classifier runs
`openai/clip-vit-base-patch32` locally (CPU here), with immutable model revision and
package versions recorded in `data/claims/graph/panorama_score_run.json`. It reuses
`experiments/sun_from_pano.py:gql` for the authenticated Matterport Model API.
Credentials are parsed only inside that helper at runtime; remote bodies and URLs
are never logged. No public viewer is scraped. Model weights and metadata live
under `data/matterport/`; generated JSON contains no signed URLs.

```sh
.venv-agent/bin/python -m pip install --no-cache-dir -r agent/requirements-panorama.txt
.venv-agent/bin/python -u agent/panorama_features.py acquire --limit 36
.venv-agent/bin/python -u agent/panorama_features.py score --limit 36
.venv-agent/bin/python agent/panorama_features.py sheet
# Review the sheet, then record explicit label thresholds in the calibration JSON.
.venv-agent/bin/python agent/panorama_features.py publish
```

Acquisition orders the six featured and 36 heavy homes first, then other indexed
homes. Increase `--limit` for a larger cohort. Run acquisition before scoring;
completed homes are reused. `acquire --beam-listings` and `score --beam-listings`
inspect the extra cached beam/rafter listing cohort. Listing text only selects
images to inspect and never establishes a visual observation. At most six API
panoramas per home are selected, covering distinct closed-boundary room polygons
first. All six cube faces are saved as 640px JPEGs. Horizontal faces get full,
upper and lower crops; ceiling/floor faces are also scored. Images are reencoded
without metadata. The classifier scores the fixed 14-label vocabulary independently;
cosine similarity is **not a probability**. A node is a crop observation, so multiple
nodes can describe the same physical feature. Hardwood means wood-like appearance,
not verified flooring material.

Review artifacts: `data/claims/graph/panorama_features_sheet.jpg` (at most 40 labelled
crops), its JSON provenance, and `panorama_features_calibration.json`. Conservative
label thresholds disable labels whose sheet examples are false or ambiguous. The
spot-check precision estimate is biased and is not verified population precision;
recall is unknown. Never rerun `sheet` over a reviewed sheet without preserving its
review provenance and obtaining another spot check.

Current run: 169 homes (all36 heavy homes plus133 additional), 1,014 panoramas,
6,084 cube faces. The expanded sheet enabled exposed beams/rafters at .339 and
disabled skylights after a tray-ceiling false positive. Other enabled cutoffs are
fireplace .340, hardwood .350, carpet .335, ceiling fan .345 and shelves .345.
All other labels remain disabled. Scores use full precision; the sheet rounds to
three decimals. The paired Claude spot-check judged all11 above-cutoff examples
correct, including six beams: an estimate from a biased sheet, not verified precision.
The original36home review sheet is preserved as `panorama_features_calibration36_sheet.jpg`.
The proof finds five rooms with beams, including one bedroom in `87jACYuvreU`;
it finds no primary-bedroom + beams + winter60-minute match. Bedroom role remains
largest_bedroom for that home; this result does not establish a primary bedroom.

Accepted sidecars are in `data/claims/graph/panorama_features/accepted_<sid>.json`;
the ordinary per-home **graph** files contain the same nodes and room-feature edges.
Rebuilding `agent.home_graph` reloads these sidecars. Original per-home claim files
are untouched. `graph_search` accepts `source_kind: "seen in panorama"`, a literal
`feature: "beam|rafter"` alternative, and optional `min_feature_confidence`. It returns
the crop, image, score, model and exact `seen in panorama <id> face <n>` attribution.
Camera-room mapping does not prove that a feature visible through an opening lies
inside that room. Unknown rooms are searchable without room/numeric constraints;
unknown rooms never satisfy sun or bedroom filters. Largest bedroom never means
primary bedroom without an explicit primary/master role or label.

`data/claims/eval/panorama_features_proof.json` records real local searches, counts,
the per-home claim hash guard and caveats. `panorama_features_input_manifest.json`
hashes the observed image/metadata inputs without copying them. The disk guard stops
below 1 GiB free or above 3 GB conservatively attributed allocation, including all
model files and relevant installed dependency directories. Concurrent workers' disk
writes are reported through free space but are not attributed to this task.

The seeded **random-sample spot check** is `data/claims/graph/panorama_random_sheet.jpg`
and `.json`; generate with `agent/panorama_features.py random-sheet --seed 20261114`
using the venv Python. Do not overwrite its completed review. It samples published
nodes uniformly within label strata: all six beams, five shelves and one fireplace,
plus 20 fans, four carpets and four wood-look floors. Claude judged visual presence
40/40 and camera-room assignment 39/40; tile 06 shows carpet beyond its hallway.
Population-weighted room-assignment precision is estimated at 97.0%; presence at
100% under wood-look semantics. Hardwood material is unestablished (0/4 confirmed).
These are estimates, not verified precision or recall. Rare strata overlap calibration;
this is not an independent held-out evaluation. Thresholds remain unchanged.
Counts, inclusion probabilities and reviewer provenance are recorded in the JSON
and `data/claims/eval/panorama_random_precision.json`.

`audit_home` gives the listing-auditor separate `panorama_evidence`, including source,
cosine confidence, model and camera-room context. Disabled labels answer
"no panorama evidence". Responses use `display_label`: raw fixed detector label
`hardwood floor` displays as `wood-look floor`, with material unverified.
The CLI replay and `agent/eval.py --dry-run` run an additional actual Deep Agents
scripted case: "a bedroom with exposed beams that gets winter sun". The recording
is `data/claims/eval/panorama_dry_run_trace.json`, with a preserved trace in each
run workspace. Current qualified homes: none. `87jACYuvreU` is a visual near match
whose room label is Bedroom (largest_bedroom), not primary; unreliable placement
withholds winter timing, so it cannot satisfy the winter-sun requirement. The
diagnostic query drops sun only to explain this exclusion, never to select a finalist.
The auditor and sun specialist execute real tools; language decisions are scripted.

```sh
.venv-agent/bin/python -I agent/agent.py --dry-run
.venv-agent/bin/python -I agent/server.py --dry-run
```

The replay uses a scripted chat model through the actual `create_deep_agent` graph: planning, filesystem, three isolated parallel listing auditors, sun analyst, verifier, draft approval, annotation approval and final answer. Recording: `data/claims/eval/dry_run_trace.json`. It proves orchestration, not model quality or human approval.

Per-thread files are in `data/claims/workspaces/<thread>/`: `needs.md`, `candidates.json`, `dossiers/<sid>.md` and `.json`, `figures/sun.png`, `comparison.md`, `questions_for_agent.md`, `viewing_plan.md`, `annotations/<sid>.json`, and the approved `approved_draft.md`. Drafts are never sent. Buyer edits are labelled as unverified wording. The verifier checks retained tool-generated content and numeric evidence, not independent geometry truth.

## OpenRouter

Add these exact lines to `.env`, replacing the key value:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
AGENT_MODEL=openrouter:inclusionai/ling-3.2-flash:free
AGENT_SUBAGENT_MODEL=openrouter:nvidia/nemotron-3.5-lightning:free
```

Run `.venv-agent/bin/python -I agent/server.py` or `.venv-agent/bin/python -I agent/agent.py --ask 'Houston under $600000, morning sun in an office'`. Only application code reads `.env`; keys/provider error bodies are not logged. OpenRouter uses its OpenAI-compatible endpoint through `init_chat_model`, with retries disabled. `data/claims/eval/openrouter_tool_models.json` records public catalog models advertising free tool calling, including these two. Authenticated inference and model quality remain unverified. Optional `LANGSMITH_API_KEY` enables tracing and live evaluation reporting; otherwise tracing is disabled.

## HTTP and demo integration

The server binds `127.0.0.1:8770` and allows local browser origins. `GET /status` returns HTTP 200 before model setup with boolean `configured`, `dry_run`, `real_model_run` fields. Dry-run counts as configured; `real_model_run` identifies configured live mode, not an inference made by the status request. Configuration is inspected without constructing or contacting a model. Other endpoints return clear HTTP 503 JSON without configuration. `--dry-run` explicitly selects the fake model.

| Endpoint | Request/result |
| --- | --- |
| `GET /status` | Always HTTP 200 JSON readiness; no model setup or inference |
| `POST /ask` | `{"question":"...","profile_id":"default"}`; NDJSON events |
| `POST /approve` | `{"thread":"...","approval_id":"...","decision":"approve"}`; resumed NDJSON |
| `POST /feedback` | `{"profile_id":"default","needs":{},"feedback":[{"sid":"...","thumb":"up","reason":"..."}]}`; preference approval |
| `POST /check` | `{"sid":"..."}`; grounded audit JSON |
| `GET /files/<thread>/<path>` | Workspace file, traversal and symlink escape rejected |
| `GET /annotations/<sid>` | Published annotation JSON |

Events include `thread`, `type`, and event-specific data: plan updates, tool calls/arguments and short results with file pointers, specialist starts/finishes, files, approval requests, final answers and sanitized errors. Read `approval_request.approval` for the id/actions. Decisions are `approve`, `edit`, `reject`. Draft edits supply `text`; preference edits supply `preferences:{"needs":{},"feedback":[]}`. Drafts, preference saves and annotation publication have separate interrupt gates. Duplicate approvals are rejected; rejecting publication leaves existing notes unchanged. Sessions/checkpoints are in memory: server restart loses pending approvals.

`agent/demo_hook.js` defines `demoAgent`, `demoAgentApprove`, `demoAgentFeedback` and an NDJSON reader with callbacks and a `demoAgentEvent` browser event. No `demo/` files were changed; its UI integration is unverified.

Preferences persist under `agent/state/<profile_id>.json` through a composite `/memories/` backend; seven skills are loaded on demand from `agent/skills/`. A later search shows the remembered reason that changed ranking. Feedback must be explicit and approved. Profiles are local identifiers, not authenticated multi-user accounts.

Annotation shape (coordinates shown are illustrative):

```json
{"sid":"...","coordinate_frame":"Matterport scan metres; x/y horizontal, z vertical","published":true,"approval_mode":"buyer_review","thread":"...","annotations":[{"id":"...","x":0.0,"y":0.0,"z":0.0,"room_id":"...","window_id":"...","title":"Light","text":"Measured explanation","verdict":"unclear","evidence_ids":["..."],"anchor_source":"data/sun/<sid>/sun.json","anchor_kind":"Measured window centre in the evidenced room; not a location of the claimed feature."}],"omitted_claims":0,"caveats":[]}
```

Anchors use recorded window centers, with lower-z plus half the height. Claims lacking an evidenced room/window anchor are omitted. Replay publication is explicitly marked `scripted_dry_run`.

Annotation publication accepts approve or reject. Editing measured notes requires revising their evidence and requesting publication again; it cannot silently approve unchanged notes under an edit decision.

## Property graph and search

Build cached JSON with `.venv-agent/bin/python -m agent.home_graph`. `data/claims/graph/` has per-home graphs, an index and an input hash manifest: home/floor/room/window/claim/feature nodes; home-room, floor-room, room-window, room-feature, claim-about-room and cached doorway room-room edges. Room nodes carry type, area, ceiling height, four dates of sun minutes and first/last lit envelopes; windows carry compass azimuth, area and sun minutes. Missing data stays unknown; unreliable placement withholds sun. Lit envelopes may include conditional doorway light and are not continuous windows-only intervals.

`graph_search(constraints)` combines structured home needs, room properties, sun and case-insensitive literal text alternatives separated by `|`:

```json
{"room":"bedroom","feature":"beam|rafter","date":"2026-12-21","min_sun_minutes":60,"limit":10}
```

Other fields: `needs`, `source_kind`, `min_area_m2`, `min_ceiling_height_m`. Features are attributed to `Matterport room label`, `listing says` or `seen in panorama`, with separate provenance. A listing clause attaches to a room only when it names one room type; primary-bedroom text requires a known room role. Listing features remain spatially unverified. The strict bedroom example currently returns empty; living-room beams have attributed matches. Empty hard requirements must not be silently dropped. The replay queries the graph during planning.

Buyer needs support city, budget, pool, ceilings, room/date/sun constraints and explicit nice-to-haves. Missing measurements never pass. After an empty search, one stated nice-to-have can be relaxed, with that change reported. Only four computed dates are available: `2026-12-21`, `2026-03-20`, `2026-06-21`, `2026-11-14`; arbitrary-date sun is unavailable. Viewing plans use November. Tree and placement caveats remain; contained partial scans are described as one unit in a larger building, not placed on the aerial photo.

## Privileges, analysis and evidence

Outside-Codex verification by paired Claude passed the revised profile: all six permission checks true, exit 0, empty stderr, no model code executed. Recorded in `data/claims/eval/sandbox_outside_proof.json`. HOME denies file data reads while preserving metadata access; explicit data-read allowances cover the venv, base interpreter, resolved Python executable, facts and workspace. The enforcement probe still runs before each model-written analysis.

Listing text is untrusted. Its auditor has no code execution and cannot write outside its dossier. The sun analyst receives numeric facts and ids, never listing remarks. The main agent has no execution tool. `run_analysis(script)` requires a sandbox enforcement self-test: Python `-I`, stripped environment, read-only numeric facts, no network, 30-second timeout, output cap. The original profile failed outside Codex too, so nesting was not an established cause. The rebuilt profile starts with allow-default, then denies network, writes except the workspace and `/dev/null`, and home file-data reads except the venv, base interpreter, resolved Python executable, facts and workspace. Metadata reads remain allowed for realpath. Facts remain read-only. The probe checks network denial, outside-write denial, `.env` open denial (never reading content), an allowed facts read and workspace write. It reports stderr and return code; Python startup failures use `sandboxed_python_failed`, while sandbox-exec launcher failures use `sandbox_apply_failed`. Model-written code stays disabled unless every check passes; the existing recording uses a trusted fixed chart. Run outside Codex:

```sh
/Users/phanan/HackPrinceton/.venv-agent/bin/python -I /Users/phanan/HackPrinceton/agent/analysis.py --self-test
```

401/403 stops subsequent tool execution. Final numbers absent from actual tool results are blocked. These checks limit unsupported claims, not semantic errors. Panorama search uses cached local classifications and reviewed crops; live model-driven visual inspection remains deferred.

```sh
.venv-agent/bin/python -I agent/prove_mcp.py
.venv-agent/bin/python -I agent/eval.py --dry-run
```

For MCP, use executable `/Users/phanan/HackPrinceton/.venv-agent/bin/python`, arguments `-I`, `/Users/phanan/HackPrinceton/agent/mcp_server.py`. Sixteen narrow tools include graph search; preference writes and publication are not exposed. The evaluation records 50 fixed listings/10 buyer requests, topic agreement, extra claims, numeric rejection, tool calls and tokens. Scripted fixtures replay rule verdicts: agreement proves the harness, not quality. `agent/eval.py --live` explicitly invokes authenticated models and may incur cost; it has not been run. Existing seven-test suite passed; no tests were added. Fresh HTTP evidence is `data/claims/eval/http_buyer_proof.json`, including approvals, preference edits/rejection, remembered ranking, file retrieval, no-config responses and unchanged per-home claim hashes.

## Preserved inputs, attribution and undo

This takeover never regenerates `data/claims/<sid>.json`. Historical rules/proof scripts are retained but running them rewrites home claims and is outside this takeover. The complete `revision_snapshot/` now uses APFS clones for 9681 byte-identical inputs; 10 differing files were preserved. `agent/disk_reclaim.json` records free space before/after. New freezes use hash manifests, never full copies. Stop below 1 GiB free.

Codex changed `agent/` and generated workspaces, graph and evaluation artifacts. No upstream inputs, demo files, commits or git initialization were changed. Source patches are snapshotted: `python3 ~/AgentBridge/source/scripts/edit-ledger.py list --path /Users/phanan/HackPrinceton/agent/buyer_agent.py`, then `rollback <id>` for dry run or `rollback <id> --yes` to undo. Preserve desired generated artifacts before regeneration. `agent/PROGRESS.md` holds continuation status.
