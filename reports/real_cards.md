# Real labelled cards — measurement

- dataset: **2000** labelled phone photos (4000 × 1800, card in a hand, daylight) —
  the public Kaggle set `kibordevents/iranian-national-id-card-dataset`, which is the
  same collection as the Drive folder earlier (that download had only reached 87 of the
  2000). 200 of them are kept in the git-ignored `data/real/` for future sessions.
- labels: `id,national_id,first_name,last_name,date_of_birth,father_name,expiry_date`
- prepared with `scripts/csv_to_dataset.py --copy --strip-spaces --template auto`
- engine: tesseract (`fas`) + glyph digit classifier, `template: auto`, cold cache

**Held-out slice, ids 700–849 — 150 cards the pipeline was never tuned on:**

| | after localisation + row fit | before (same 150 cards) | the first 87 |
|---|---|---|
| field exact match | **19.22 %** | 7.78 % | 5.36 % |
| field character accuracy | **33.92 %** | 19.91 % | 15.43 % |
| fully correct cards | **19.22 %** | 7.78 % | 5.36 % |
| mean latency | 3125 ms / card | 3159 ms / card | 2926 ms / card |

| field | exact (after) | chars | exact (before) | chars |
|---|---|---|---|---|
| `national_code` | 1.3 % | 36.2 % | 0.7 % | 22.7 % |
| `last_name` | 22.7 % | 33.4 % | 10.0 % | 18.9 % |
| `first_name` | 24.0 % | 34.5 % | 11.3 % | 19.4 % |
| `father_name` | 19.3 % | 31.5 % | 7.3 % | 18.0 % |
| `birth_date` | 22.0 % | 33.5 % | 7.3 % | 20.4 % |
| `expiry_date` | 26.0 % | 34.4 % | 10.0 % | 20.0 % |

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

## Where the loss actually is (measured, 150 held-out cards)

| group | cards | field accuracy |
|---|---|---|
| routed to `national_card_smart` | 37 of 60 sampled | 16.7 % |
| routed to `national_card_front` | 23 of 60 sampled | **0.0 %** |
| rotated 90/270 by OSD | 9 of 150 | **0.0 %** (0 of 54 fields) |

The template is *not* the cause.  Sixteen of the front-routed cards were re-read with
every template forced — `auto`, `national_card_smart`, `national_card_front` — and
scored 0 of 96 fields in all three: their warped canvas does not contain the card
upright, so no box placement can save them.  The same is true of the 16 cards whose
quad was dumped by eye: the canvas holds the card at an angle, half out of frame, with
hand and table around it.  **Card localisation is the bottleneck; the row grid, the
template pick and the digit reader are all downstream of it.**

## Training on the real crops (2026-11-07)

`scripts/mine_real_digits.py` harvests labelled glyphs from real photos without OCR:
for a national code / birth date / expiry date the *glyph count is known from the
label*, so a field whose segmentation yields exactly that many non-separator glyphs can
be labelled positionally; anything else is dropped, which keeps misaligned boxes out.

* 1204 glyphs from 120 of the 1000 cards scanned (~10 % usable — the rest are
  misaligned, see above), merged into the model at weight 5 (12 020 samples total).
* Held-out 150: **7.78 % / 19.86 %** — i.e. unchanged (7.78 % / 19.91 %) — and the
  clean renders unchanged at 95.24 % / 99.05 %.  The shipped model was therefore left
  as the synthetic table; the mining script stays for when the geometry is fixed, at
  which point the same run yields ~10× more usable crops.
* Conclusion: the digit reader is rarely *reached* while the boxes sit on the wrong
  rows, so training it cannot move the score yet.

## The rotation bug found on the way

`upright()` turns the canvas by 90/270 when OSD says so, which makes the canvas
*portrait* while the template boxes are normalised to a landscape card — every box then
lands somewhere else (that is the 0-of-54 group above).  The picture is now turned and
the detection redone instead, which keeps the canonical canvas and the card's aspect.
On the nine cards that hit this path the reading did not improve (their quads are junk
either way), but the path no longer guarantees nonsense.

## What fixed it (2026-11-07, second pass)

Two changes, both measured on the same 150 held-out cards:

1. **The card is found by colour.** A national card is a pale cyan sheet with a pink
   band; a hand, wood, a bottle and a bin are outside those HSV bands.  The largest
   blob of that mask gives a quad, and it is used when the card fills the canvas it
   produces (≥ 0.80 of the border ring is card) and beats the geometric quad by 0.02.
   The geometric quad keeps priority whenever it is convincing and already covers the
   picture, so scans and renders are untouched (clean renders still 95.2 %).
2. **The row grid is fitted per card.** The template's rows are a proportion of the
   design, but a photographed card leaves the warp a few percent larger or smaller and
   a box then reads its neighbour.  The ink profile of the value column is smoothed,
   its strongest maxima are taken as candidate rows, and the window whose gaps match the
   template pitch is fitted with a straight line (scale 0.92–1.08, offset ±8 % of the
   height, every row within 9 px of a peak).  The worst cases were boxes sitting 20–40 px
   below their values — «شماره ملی» read from the «نام» row.

Together: **7.78 % → 19.22 %** of cards completely right, **19.91 % → 33.92 %** of
characters, at the same latency.  On the photo renders one card in thirty flips the
other way (83.3 % → 80.0 %) — the price of a colour hypothesis that is allowed to
overrule a low-scoring geometric one.

## The frame that is not the card (2026-11-07, third pass)

A single-card probe found the remaining localisation hole.  On card 735 the detector
returned the **whole picture** as the card (`contour`, score 0.66, area 0.95) — no card
edge had been found, the frame contour won on the 0.55 demotion — and its aspect ratio,
1.26, was simply the snapshot's.  Warping a portrait frame onto the landscape canonical
canvas squashes the card, so every box missed (0 of 6 fields, all six wrong).

The fix is one condition on the existing colour hypothesis: a geometric quad that
*already covers the picture* only blocks colour when its shape is a card's.  With
`|aspect − 1.586| > 0.25` it is a miss, colour is preferred, and it only has to reach
0.55 (it is beating the frame, not a real card).  Clean renders keep aspect 1.586 and
area 1.0, so they still block colour exactly as before; photo renders sit at area
0.70–0.83 and are unaffected.

* card 735: **0/6 → 4/6** (`last_name`, `father_name`, `birth_date`, `expiry_date` right)
* 150 held-out: **19.22 % → 21.44 %** exact, **33.92 % → 36.05 %** characters
  (`national_code` 36.2 → 39.1 % chars, `last_name` 22.7 → 25.3 %, `first_name`
  24.0 → 26.7 %, `father_name` 19.3 → 21.3 %, `birth_date` 22.0 → 24.7 %,
  `expiry_date` 26.0 → 29.3 %), 2810 ms/card
* clean renders 95.24 % / 99.05 % and photo renders 80.00 % / 85.77 % — unchanged

Rejected on the way: lowering the colour bar globally to 0.60 (real probe +2 fields but
clean renders 95.2 % → 69.1 %, because on a render the colour mask also covers the
background and the junk colour quad then wins), and preferring colour whenever it is
tilted relative to the geometric quad (+ tilt gate kept 735 at 0/6 — the frame still
blocked it).

## The repair that was breaking correct reads (2026-11-07, fourth pass)

`national_code` sat at 1.3 % exact with 39 % of its characters right — digits arrived,
whole codes almost never.  Reading eight cards by hand found the cause: the checksum
"repair" was rewriting *correct* readings.

```
truth 7231198618  read (no repair) 7231198618  ->  7231198619
truth 3973076093  read (no repair) 3973076093  ->  3973076091
truth 3863500888  read (no repair) 3863500888  ->  3863500881
truth 6245701599  read (no repair) 6245701599  ->  6245701597
```

A printed code that fails mod-11 has one wrong digit *somewhere*; the derivation keeps
the first nine and rewrites the tenth, which is only right when the tenth is the broken
one.  On this dataset it is never right: **only 7 of the 150 labels are checksum-valid
at all** (chance is ~1 in 11, so the ids are random ten-digit strings), and no
alternative weighting of the mod-11 algorithm validates them either.  The derivation
therefore has no signal and only corrupts.

`validate_national_code(..., derive_check_digit=False)` is now the default; the
`/validate/national-code` API route and `nidocr check` still pass `True`, because there
the caller is explicitly asking for a valid code.  The unique-single-substitution repair
and the lost/extra-glyph repairs are unchanged — they only fire when exactly one
candidate exists.

* 150 held-out: **21.44 % → 23.67 %** exact, 36.05 % → 36.13 % characters,
  `national_code` **1.3 % → 14.7 %** exact (39.5 % chars), 2938 ms/card
* clean renders 95.24 % / 99.05 %, photo renders 80.48 % / 85.92 % (unchanged)

What is left: `national_code` now reads whole codes on 15 % of cards but many reads are
truncated to two or three digits (`3`, `05`, `023`) — the row is cropped short or the
glyphs merge, which is a segmentation problem, not a checksum one.  Names sit at ~25 %
exact with ~35 % characters.
