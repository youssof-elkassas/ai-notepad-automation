# ScreenSeekeR Notepad Automation — Design Document

## 1. Assumptions

- **OS:** Windows 10 or 11 (primary monitor, English UI recommended).
- **Resolution:** 1920×1080 at 100% display scaling (higher DPI scales are detected and logged).
- **Hardware:** NVIDIA GPU with 8–16 GB VRAM recommended; low-spec profile uses 4-bit quantization and smaller models.
- **Desktop:** A Notepad shortcut/icon is visible on the desktop.
- **Network:** Internet access for JSONPlaceholder API and HuggingFace model download on first run.
- **Models:** Open-source substitutes for paper's GPT-4o planner (Qwen2.5-VL) and OS-Atlas-7B grounder.

## 2. Architecture

The system implements the **ScreenSeekeR** agentic grounding framework from [ScreenSpot-Pro (arXiv:2504.07981)](https://arxiv.org/pdf/2504.07981):

```
Screenshot → GUI Preprocess → Planner Position Inference → OS-Atlas Grounding
    → Patch Score/NMS → Recursive Crop → Planner Result Checking → Mouse/Keyboard Action → Verify
```

Perception (vision) is fully separated from automation (mouse/keyboard). The pipeline never reuses coordinates across iterations.

## 3. Grounding Pipeline

1. **Observe:** Capture fresh 1920×1080 screenshot via `mss`.
2. **Position Inference:** Qwen2.5-VL planner proposes candidate desktop regions (paper Appendix C prompts).
3. **Ground:** OS-Atlas predicts bounding boxes per region (0–1000 normalized coords).
4. **Score & NMS:** Gaussian centrality scoring (paper Eq. 1–2) ranks candidate patches.
5. **Recursive Search:** Crop into top patches until patch ≤ 1280 px, then direct grounding.
6. **Verify:** Planner checks annotated crop (`is_target` / `target_elsewhere` / `target_not_found`).
7. **Act:** Double-click center, type content, save file, close Notepad.
8. **Repeat** for next post with a new screenshot.

**Low-spec fallback:** ReGround (1024×1024 crop around initial prediction) per paper §4.1.

## 4. Model Selection

| Role | Paper | This Implementation | Rationale |
|------|-------|---------------------|-----------|
| Grounder | OS-Atlas-7B | OS-Atlas-7B / 4B | Best specialist on ScreenSpot-Pro (18.9% base) |
| Planner | GPT-4o | Qwen2.5-VL-7B / 3B | Open-source; strong screenshot understanding |
| GUI Parsing | Planner hierarchy | `gui_parser.py` preprocessing | Coordinate transforms + annotation |

Rejected: template matching, OCR-only, GroundingDINO (not GUI-tuned), Florence-2 (weak on tiny icons).

## 5. Tradeoffs

| Choice | Benefit | Cost |
|--------|---------|------|
| ScreenSeekeR vs single-shot | +29% accuracy on benchmark | 3–10× inference latency |
| Local models vs API | Privacy, reproducibility | VRAM requirements |
| Keyboard Save-As vs clicking dialog | No hardcoded dialog coords | Fragile to Notepad UI changes |
| pywinauto verification | Reliable window detection | Windows-only |

## 6. Failure Cases

- **DPI scaling ≠ 100%:** Coordinate drift; mitigated by DPI detection and warnings.
- **Overlapping desktop icons:** Planner neighbor inference may pick wrong region.
- **Modal dialogs / popups:** Obscure target; pipeline saves failure screenshot and exits.
- **Localized Windows UI:** Save dialog shortcuts may differ; configurable paths.
- **Low VRAM:** OOM on 7B models; use `--profile low`.

## 7. Scalability

The generic `GroundingService.locate(instruction, screenshot)` API supports arbitrary elements ("Chrome", "Save button", "OK") without code changes — only the instruction string changes.

Adding new applications requires: new pipeline steps (not new grounding code), state verification hooks, and keyboard/mouse action sequences.

## 8. Performance (Estimates)

| Profile | Grounding Latency | VRAM |
|---------|-------------------|------|
| High (7B+7B) | 30–90 s/search | ~14–20 GB |
| Low (4B+3B INT4) | 60–180 s/search | ~6–8 GB |
| ReGround fallback | 10–30 s | Same as grounder |

Full 10-post run: ~15–45 minutes depending on profile and hardware.

## 9. Future Improvements

- OmniParser as validation layer (cross-check grounder bbox against detected elements).
- UI-TARS or GTA1 grounder upgrade for higher base accuracy.
- Active popup detection and dismissal agent loop.
- vLLM serving for batched inference.
- Multi-monitor support with monitor selection config.
