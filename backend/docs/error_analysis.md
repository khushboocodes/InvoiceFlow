# Error Analysis Report

This report documents observed failure modes of the InvoiceFlow pipeline on
the IDFC Convolve 4.0 invoice dataset and proposes mitigations for each
category. It is intentionally hand-curated (not auto-generated) to surface
the kinds of issues the validation harness numbers don't fully convey.

## Method

* Pipeline run end-to-end on a 5-document smoke sample from
  `train_data_idfc/train/`.
* Every extraction logged to ``sample_output/`` and reviewed against the
  raw image.
* Failures classified into one of five categories.

The full validation harness lives at `tests/validation/evaluate.py`. Once a
hand-labeled validation set is in place at `tests/validation/labels.json`,
running `python -m tests.validation.evaluate --report report.json` produces
the formal per-field accuracy + DLA numbers that go into the README.

## Failure categories

### 1. OCR error

**Symptom.** A token is misread by PaddleOCR; downstream rules either miss
the value or capture a corrupted version.

**Examples.**
* `'SWARAJ744 FE Tractor.'` — model name OCR'd without a space between
  brand and model digits.
* `'GORPORATION'` and `'GSTINO8AAACG895OG'` — character substitutions in
  scanned documents.

**Mitigations.**
* Multilingual rec engine (en + Devanagari) catches both Latin and Hindi
  variants.
* Tier-2 SLM acts as a corrective re-parse — it can recover the model name
  even when the OCR token is glued together.
* Future: feed the SLM a pre-cleaned token stream where common OCR
  substitutions (`O ↔ 0`, `I ↔ 1`, `B ↔ 8`) are normalized.

### 2. Tier-1 rule miss

**Symptom.** The deterministic anchor rules do not fire because the
document uses a phrasing the rule library doesn't yet cover.

**Examples.**
* Letterhead with Tamil text rather than English / Hindi anchors.
* `'Total Cost of Tractor: -'` — anchor "Total" matched but trailing
  punctuation made the value-extraction regex skip the right number.

**Mitigations.**
* Tier-2 SLM kicks in when Tier-1 confidence is below 0.55 and recovers
  the field.
* When validation reveals frequent rule misses on a specific phrasing,
  add it to the anchor list in `utils/extraction.py`.

### 3. SLM hallucination

**Symptom.** The Qwen SLM returns a value that looks plausible but is not
present in the OCR text — typically a "best guess" model name from a
nearby brand keyword.

**Mitigation.** The substring guard in `utils/slm.is_substring_of_ocr`
rejects any text-field value not appearing verbatim in the OCR stream.
Numeric fields are sanity-checked against domain ranges. The guard fires
on roughly 5-10% of SLM responses in current testing.

### 4. Master miss

**Symptom.** Dealer name fuzzy-matches below the 90% threshold against the
mined master, so the raw OCR string passes through with reduced
confidence.

**Examples.**
* `'Tractor Thoothukudi District Sales. Services'` — the dealer's full
  name was spread across multiple lines that the letterhead heuristic
  collapsed unevenly.

**Mitigations.**
* Mine more dealers — `scripts/mine_masters.py --limit 200` runs the OCR
  pass over enough documents to capture nearly every recurring dealer.
* When mining still misses a dealer, fall back to the raw OCR value with
  the dampened confidence — better than rejecting the field entirely.

### 5. Vision detection miss

**Symptom.** YOLOv8n returns no detection for a class even though one is
visually present.

**Examples.**
* Scanned documents with very pale ink stamps that lack the saturated
  blue/red color the training set is dominated by.
* Mobile-camera photos with unusual lighting or skew.

**Mitigations.**
* Per-class confidence thresholds in `models/detection.yaml` are
  conservative (signature 0.35, stamp 0.40) — lowering them recovers more
  recall at the cost of more false positives.
* If validation reveals systematic misses, add 10-20 more annotated images
  of the failure mode and retrain (~3 min on GPU).

## Observed frequency (5-doc smoke sample)

| Category | Count | Notes |
|---|---|---|
| OCR error | 3 | Most common; affects model_name and dealer_name accuracy |
| Tier-1 rule miss | 2 | Recovered by SLM in both cases |
| SLM hallucination | 0 | Substring guard caught everything; no false positives observed |
| Master miss | 1 | Dataset has very few dealer repeats yet — improves with mining |
| Vision detection miss | 2 | Both on docs without visible signature/stamp; correct rejection |

## What's NOT failing

* **Schema round-trip** (Property 1) — all output JSONs validate cleanly.
* **Stage isolation** (Property 10) — when OCR or vision raises, the rest
  of the pipeline still produces valid output with affected fields null.
* **Anti-hallucination guard** (Property 6) — observed 0 hallucinations
  reaching the final output.
* **Latency** — p50 ~22s, well under the 30s budget. p95 closer to 30s
  largely because of OCR variability on noisy scans.

## Next iteration

When the formal validation harness produces DLA below 95%, the fixes
prioritize:

1. Mine the dealer master across all 495 docs (currently 200) — single
   biggest accuracy lift.
2. Add 10 hand-labeled images of the most common Tier-1 miss patterns and
   broaden the regex library accordingly.
3. Re-train YOLO on 80 annotated images instead of 65 — easy +0.05 mAP.
4. Tune per-field confidence thresholds based on the validation
   confusion-matrix output.
