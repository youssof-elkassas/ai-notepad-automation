# Interview Preparation — ScreenSeekeR Grounding

## Why this grounding approach?

ScreenSpot-Pro (arXiv:2504.07981) demonstrates that specialist GUI grounding models (OS-Atlas) struggle on full-resolution professional desktops because targets occupy ~0.07% of screen area. ScreenSeekeR solves this by combining:

1. **Planner GUI knowledge** — reasons about where UI elements likely live (desktop left, taskbar, etc.).
2. **Cascaded visual search** — progressively crops the screenshot to reduce distraction.
3. **Specialist grounder** — OS-Atlas operates on smaller, focused patches where it is most accurate.

This mirrors human visual search: scan broadly, narrow focus, confirm target.

## Why not template matching?

- Breaks when icon theme, resolution, or wallpaper changes.
- Requires prior knowledge of icon appearance (explicitly forbidden).
- Cannot generalize to "Save button" or arbitrary controls.
- Fails on Windows 11 fluent icons vs Windows 10 style.

## Why not OCR?

- Many desktop icons have text labels below the icon, not in the clickable region.
- OCR finds text but not the interactable bounding box.
- Buttons like toolbar icons have no text.
- OCR cannot ground "the third icon in the second row" semantically.

OCR can supplement (as OmniParser does) but cannot be the primary grounding mechanism.

## Failure cases

1. **Tiny targets in cluttered screens** — mitigated by ScreenSeekeR cropping.
2. **Planner proposes wrong region** — recursive search explores alternatives; ReGround fallback.
3. **Grounder hallucination** — planner verification rejects with `target_elsewhere`.
4. **DPI scaling** — coordinate mapping errors; detect and warn.
5. **Dynamic UI** — notifications, tooltips covering target.
6. **Similar-looking icons** — neighbor inference confusion.

## Performance analysis

- **Bottleneck:** VLM inference (2 models × multiple crops per search).
- **ScreenSeekeR depth 3:** up to 5 candidates × 3 depths = worst case 15 grounder calls.
- **Optimization paths:** INT4 quantization, ReGround-only on low-spec, planner on downscaled overview.
- **Compared to template matching:** 100× slower but infinitely more general.

## Scaling to arbitrary desktop applications

1. Change `icon_instruction` to any natural language target.
2. Add application-specific action sequences in `pipeline.py` (typing, saving, clicking).
3. Reuse the same `GroundingService.locate()` for every UI interaction.
4. Add state verification per application (window titles, file existence).

The grounding module is application-agnostic by design.

## Handling unknown popups

Current behavior: grounding may fail or planner returns `target_not_found`; failure screenshot saved; graceful exit.

Future: add an Observe loop that detects unexpected modal dialogs via planner ("describe this screenshot — is there a blocking dialog?") and dismisses via grounded "OK"/"Close" before retrying.

## Potential improvements

1. **Bidirectional zoom** (MEGA-GUI) — recover from early search errors.
2. **OmniParser validation** — cross-check bbox against detected interactable set.
3. **UI-TARS-72B grounder** — higher base accuracy on ScreenSpot-Pro leaderboard.
4. **vLLM batch serving** — amortize model load across crops.
5. **Experience replay** — log successful search traces for debugging, not for coordinate reuse.
