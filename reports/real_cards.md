# Real labelled cards — measurement

- dataset: private, **87** phone photos (4000 × 1800, card in hand, daylight), labels from
  `labels.csv` (`id,national_id,first_name,last_name,date_of_birth,father_name,expiry_date`)
- prepared with `scripts/csv_to_dataset.py --copy --strip-spaces --template auto`
- engine: tesseract (`fas`) + glyph digit classifier, `template: auto`, cold cache

| | value |
|---|---|
| field exact match | **5.36 %** |
| field character accuracy | **15.43 %** |
| fully correct cards | **5.36 %** |
| mean latency | 2926 ms / card |

| field | exact match | char accuracy | mean confidence |
|---|---|---|---|
| `national_code` | 0.0 % | 17.7 % | 79.8 |
| `last_name` | 6.9 % | 13.9 % | 37.0 |
| `first_name` | 6.9 % | 17.0 % | 37.7 |
| `father_name` | 3.4 % | 12.7 % | 33.2 |
| `birth_date` | 8.0 % | 20.6 % | 75.9 |
| `expiry_date` | 6.9 % | 10.7 % | 40.1 |

Same pipeline on the synthetic sets: clean 93.33 % / 97.62 %, photo 83.33 % / 89.11 %.

## What the failures are (inspected, not guessed)

1. **The row grid, not the warp.** The warped canvases are upright and readable; the
   template's row boxes are what is off. A camera photo warps the card up to ±15 %
   larger than the design, so every row sits ~20–50 px from where the template expects
   it and a box reads its neighbour: `national_code` comes back as «010» (the «نام»
   label row), `first_name` as the birth date. This is the single dominant cause.
2. **Template pick** (`national_card_front` chosen for smart cards) on ~7 of 20 inspected
   cards — the ink-fraction score fixed most of these; the remaining ones follow from (1),
   because the photo box it measures is itself shifted.
3. **Digit confusions** at the end of the national code (`5116168397` → `5116168395`,
   `7234387036` → `7234387031`) and a false 180° rotation on 1 of 20 (OSD).
4. Bright/glare handling is *not* the bottleneck on this set: the clear photos are the
   ones that reach 5/6 fields; blown-out crops are handled by the existing ROI path.

## Next

* Calibrate the row grid per design from these 87 cards (median row position) instead of
  per-card fitting — the per-card fit is what the `_fit_rows` experiments kept getting
  wrong (a plausible-but-wrong window wins on background ink).
* Re-check the OSD 180° decision with the card's own text layout (the expiry row is the
  bottom row; a flipped card puts it on top).
* Re-train / re-validate the glyph digit classifier on the real crops once (1) is fixed.
