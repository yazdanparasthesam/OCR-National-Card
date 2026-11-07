# Real labelled cards — measurement

- dataset: **2000** labelled phone photos (4000 × 1800, card in a hand, daylight) —
  the public Kaggle set `kibordevents/iranian-national-id-card-dataset`, which is the
  same collection as the Drive folder earlier (that download had only reached 87 of the
  2000). 200 of them are kept in the git-ignored `data/real/` for future sessions.
- labels: `id,national_id,first_name,last_name,date_of_birth,father_name,expiry_date`
- prepared with `scripts/csv_to_dataset.py --copy --strip-spaces --template auto`
- engine: tesseract (`fas`) + glyph digit classifier, `template: auto`, cold cache

**Held-out slice, ids 700–849 — 150 cards the pipeline was never tuned on:**

| | 150 held-out cards | the first 87 (tuned on) |
|---|---|---|
| field exact match | **7.78 %** | 5.36 % |
| field character accuracy | **19.91 %** | 15.43 % |
| fully correct cards | **7.78 %** | 5.36 % |
| mean latency | 3159 ms / card | 2926 ms / card |

| field | exact (150 held-out) | chars | exact (87) | chars |
|---|---|---|---|---|
| `national_code` | 0.7 % | 22.7 % | 0.0 % | 17.7 % |
| `last_name` | 10.0 % | 18.9 % | 6.9 % | 13.9 % |
| `first_name` | 11.3 % | 19.4 % | 6.9 % | 17.0 % |
| `father_name` | 7.3 % | 18.0 % | 3.4 % | 12.7 % |
| `birth_date` | 7.3 % | 20.4 % | 8.0 % | 20.6 % |
| `expiry_date` | 10.0 % | 20.0 % | 6.9 % | 10.7 % |

Same pipeline on the synthetic sets: clean 93.33 % / 97.62 %, photo 83.33 % / 89.11 %.

## What the failures are (inspected, not guessed)

1. **The card quad, and then the row grid.** On a card whose detection is good the
   canvas is upright and readable and the boxes read the neighbouring row (a national
   code comes back as «010», the «نام» label row) — the row grid is off by ~20–50 px
   because the warp is not exactly on the card. On many cards it is worse than that:
   the warped canvas still contains the *hand and the table* (`detection.method` =
   `contour` on a 0.26–0.6 area ratio, 13–77 % of the border ring is background), so
   the card occupies part of the frame and every template coordinate is meaningless.
   Measured on 24 held-out cards: median 23 % of the canvas border is background.
   This — not glare — is the dominant cause.
2. **Template pick** (`national_card_front` chosen for smart cards) on ~7 of 20 inspected
   cards — the ink-fraction score fixed most of these; the remaining ones follow from (1),
   because the photo box it measures is itself shifted.
3. **Digit confusions** at the end of the national code (`5116168397` → `5116168395`,
   `7234387036` → `7234387031`) and a false 180° rotation on 1 of 20 (OSD).
4. Bright/glare handling is *not* the bottleneck on this set: the clear photos are the
   ones that reach 5/6 fields; blown-out crops are handled by the existing ROI path.

## Next

* **Refine the card quad.** After the warp, check that the card fills the frame (the
  border ring must be card, not hand/table) and, when it does not, re-fit the quad to
  the bright, low-saturation card region before anything else runs. 2000 labelled
  photos make this measurable in an afternoon: `csv_to_dataset` + `benchmark.py`.
* Then calibrate the row grid per design (median row position over the dataset) instead
  of per-card fitting — the per-card fit is what the `_fit_rows` experiments kept getting
  wrong (a plausible-but-wrong window wins on background ink).
* Re-check the OSD 180° decision with the card's own text layout (the expiry row is the
  bottom row; a flipped card puts it on top).
* Re-train / re-validate the glyph digit classifier on the real crops once (1) is fixed.

## Detection experiments (2026-11-07)

* **Content-based quad selection — rejected.** Scoring every hypothesis (contour, mask,
  full frame) by what its warp contains (portrait in the photo box, six text rows in the
  value column) and keeping the best *looked* right on the real photos but cost the
  photo-style renders 83.3 % → 72.4 % of cards: on a render the contour is already
  exact, and a 340-px preview cannot tell a slightly-tight crop from a good one.
* **Colour-based rescue — kept.** Where the geometric detection gives up
  (`full_frame_fallback`, 15 of 60 held-out cards) a bright/low-saturation blob is tried
  as a card quad, and accepted only if its warp scores ≥ 0.55. It fires on 1 card in 60
  and moved the held-out set 7.67 % → 7.78 % / 19.1 % → 19.9 %: real but small.
* **Lowering the shape cut-off to 0.38 — rejected.** It turns 25 of 60 "not found"
  cards into contour quads, but the held-out set went 7.78 % → 7.67 % / 19.9 % → 19.0 %:
  those quads are wrong, and a wrong quad is not better than the full frame.
* **Row growth on a scan or a render — rejected.** Growing a clipped ROI pulls the
  «نام پدر» label into the father-name value on a clean render (95.2 % → 93.3 % on the
  clean set). Growth is now limited to cards that were cut out of a larger picture.
