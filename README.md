# NIDOCR — Persian OCR for the Iranian National Card

**خواندن خودکار کارت ملی هوشمند ایران** — OpenCV finds and rectifies the card, a Persian
OCR engine + a dedicated Persian-digit classifier read every printed field, and
validators (mod-11 checksum, Jalali calendar, a 17 500-name lexicon) turn raw OCR
into trustworthy values.

```
$ nidocr read card.jpg

 کارت ملی هوشمند  ·  1337 ms  ·  tesseract
 ✓ شماره ملی        ۱۳۴۴۰۰۵۹۸۵   95.7%  رقم کنترل معتبر است
 ✓ نام خانوادگی     بیات          74.6%  نام در فرهنگ واژگان یافت شد
 ✓ نام              نازنین        81.3%  نام در فرهنگ واژگان یافت شد
 ✓ نام پدر          ابراهیم       85.1%  نام در فرهنگ واژگان یافت شد
 ✓ تاریخ تولد       ۱۳۴۰/۰۵/۲۸   96.5%  تاریخ معتبر است
 ✓ شماره سریال      ۱۱۳۲۶۱۲     100.0%  قالب معتبر است
 ✓ جنسیت            زن            88.6%  مقدار معتبر است
```

Three front ends share one pipeline:

| | |
|---|---|
| 🖥️ **Streamlit UI** | modern, RTL, self-contained (`nidocr ui`) |
| 🚀 **FastAPI service** | one route per field (`nidocr serve`) |
| ⌨️ **CLI** | `nidocr read / field / check / bench / doctor` |

---

## Table of contents

1. [Accuracy](#accuracy)
2. [How it works](#how-it-works)
3. [Install](#install)
4. [Streamlit UI](#streamlit-ui)
5. [FastAPI](#fastapi)
6. [CLI](#cli)
7. [Python API](#python-api)
8. [Templates & new card revisions](#templates--new-card-revisions)
9. [Configuration](#configuration)
10. [Project layout](#project-layout)
11. [Tests](#tests)
12. [Limitations & roadmap](#limitations--roadmap)
13. [Data sources, licence](#data-sources--licence)

---

## Accuracy

Accuracy is **measured, not claimed**. `scripts/benchmark.py` renders synthetic cards
with ground truth (a photo-style degradation model: perspective, shadows, glare,
blur, noise, JPEG) and scores the pipeline field by field:

```bash
python scripts/benchmark.py --generate 30 --difficulty clean --out reports/benchmark_clean.md
python scripts/benchmark.py --generate 30 --difficulty photo --out reports/benchmark_photo.md
```

Have your own labelled cards? A folder of images plus **one CSV** is enough — the labels
are matched by column name (English or Persian), rows whose image is missing are reported,
and the result is a dataset the benchmark and the «دقت» tab both understand:

```bash
python scripts/csv_to_dataset.py --csv labels.csv --images cards/ --out data/real
python scripts/benchmark.py --dataset data/real --difficulty custom
```

Field **exact match** / **character accuracy** — 30 cards each, 7 fields per card,
Tesseract 5.4.1 (`fas.traineddata`, `tessdata_fast`) on a 4-core CPU:

| field | clean (exact) | clean (chars) | photo (exact) | photo (chars) |
|---|---|---|---|---|
| `national_code` | **100.0 %** | 100.0 % | 80.0 % | 86.3 % |
| `last_name` | **100.0 %** | 100.0 % | 86.7 % | 89.3 % |
| `first_name` | 93.3 % | 98.3 % | 90.0 % | 90.8 % |
| `father_name` | 93.3 % | 98.6 % | 90.0 % | 92.9 % |
| `birth_date` | 83.3 % | 97.3 % | 63.3 % | 83.3 % |
| `serial_number` | 96.7 % | 99.0 % | 73.3 % | 81.1 % |
| `gender` | **100.0 %** | 100.0 % | 76.7 % | 76.7 % |
| **overall** | **95.2 %** | **99.1 %** | **80.0 %** | **85.8 %** |

Full reports: [`reports/benchmark_clean.md`](reports/benchmark_clean.md),
[`reports/benchmark_photo.md`](reports/benchmark_photo.md),
digit-classifier report [`reports/digits.md`](reports/digits.md).

How to read it:

* **character accuracy ≥ 97.8 % on flat scans**, i.e. ≤ 2 characters wrong per card —
  the values are safe to *review*, and the validator flags what it doubts.
* **national code and gender are effectively exact**: the mod-11 checksum and the
  two-word vocabulary make them self-correcting.
* **photographed cards are harder** (down to 0.62× resolution, blur σ≈0.9, JPEG 45–70
  and glare in the benchmark): 80 % of cards come out perfect, 86 % of characters
  right — and they survive a *sideways or upside-down* photo, which used to score 0. Low-res dates and serials are where it hurts — see
  [roadmap](#limitations--roadmap).
* Latency: **1.3–2.6 s per card** on CPU (single Tesseract binary, 8–14 subprocess
  calls per card; the UI and API reuse a warm process-wide pipeline).

### Real cards, not renders

The synthetic benchmark is a *proxy*. The same numbers are measured on the public Kaggle
set [`kibordevents/iranian-national-id-card-dataset`](https://www.kaggle.com/datasets/kibordevents/iranian-national-id-card-dataset)
— **2000 labelled phone photos** (4000 × 1800, the card held in a hand, shot in daylight)
— with `scripts/csv_to_dataset.py` + `scripts/benchmark.py --dataset`. On a held-out
slice of 150 cards the gap is honest:

| | clean renders | photo renders | 150 held-out real photos |
|---|---|---|---|
| field exact match | 95.2 % | 83.8 % | **24.1 %** |
| field character accuracy | 99.1 % | 89.3 % | **36.7 %** |

Real cards are a different problem: on many of these photos the card is *not* isolated
by the warp — the warped canvas still holds the hand and the table, so the card fills
part of the frame and every template coordinate is meaningless (median 23 % of the
canvas border is background on a 24-card sample). Where the quad is right, the card is
warped a little larger than the design and a box reads its neighbour row (a national
code coming back as «010» is «نام» from the row above).
[`reports/real_cards.md`](reports/real_cards.md) records the measurement, the failure
breakdown and what is left to do. 200 of the photos and their labels are kept in the
git-ignored `data/real/` — they are personal data and are **not** in this repository.

Every value is returned with `confidence` (0–100), `validated`, `repaired` and a
Persian `message`, so an application can decide to auto-accept or ask a human.

### Accuracy per field and overall — in the UI and in code

The measurement lives in the package, not in a notebook: [`nidocr/accuracy.py`](nidocr/accuracy.py)
is the single implementation used by the UI, the CLI and `scripts/benchmark.py`, so
the number in a report and the number on screen can never drift apart.

```python
from nidocr.accuracy import evaluate, load_dataset, run_synthetic, estimate_card

report = run_synthetic(count=10, difficulty="clean")        # renders + reads + scores
print(report.exact, report.characters, [r.as_row() for r in report.fields])
report.to_markdown(title="nightly")                          # -> reports/…md
```

The Streamlit app shows accuracy in two honest flavours:

| | where | what you get |
|---|---|---|
| **اندازه‌گیری‌شده** (measured) | «📊 سنجش دقت مدل» expander | real exact-match + character accuracy **per field and overall**, on synthetic cards (any difficulty/seed/count) or on your own folder of image + ground-truth JSON; downloadable as Markdown/JSON, with the list of actual misreads |
| **تخمینی** (estimated) | «🎯 دقت این کارت» tab of every card | per-field and overall estimate with **no ground truth needed**: the prior is the committed benchmark rate of that field, adjusted by the card's own confidence, validator result and repair flag — always labelled «تخمینی» |
| **اندازه‌گیری این کارت** | same tab, form | type/confirm the true values (pre-filled with what was read) and get the exact per-field ticks plus the card's exact-match and character accuracy |

Nothing is inflated: the measured panel reports 95.2 % / 99.1 % on clean renders and
80.0 % / 85.8 % on photo-style cards — the numbers in the table above.

### Rotated and mirrored photos

A phone photo is very often not upright: EXIF orientation, the camera held sideways, or
a card lying upside down. Two things make that work:

* **the warp never mirrors.** A portrait quad is turned into the canvas with a cyclic
  shift of its corners (a rotation); reversing them instead produces a *mirrored* card,
  which no OCR can rescue. This was a real bug: sideways cards scored 0/7.
* **the canvas is straightened with OSD.** One cheap Tesseract OSD call names the
  clockwise correction (`Rotate: 90/180/270`), the canvas is turned, and the exact
  angle is reported in `quality.rotation_degrees` and in the Persian warnings. When
  the reading still looks broken, `nidocr/vision/orient.py` probes all four
  orientations with a one-pass OCR score and re-reads once if another one is better.

Small photos are supersampled before reading (a 470-px card stretched to 1012 px loses
exactly the stroke information digit segmentation needs), with a second read at full
magnification when the first one comes back weak.

### Names are read character by character — the lexicon never rewrites them

A word model is a bad judge of a *name*: tesseract's Persian dictionary and the bundled
name lexicon both prefer words they have seen, and that is how a card's «شرجی» became
«سرخی» — the pixels were right and the dictionary "corrected" them. The lexicon is now
a confidence hint only: it can confirm a reading, never replace it.

Name fields are read the way the numeric ones already were — character by character
(`nidocr/ocr/char_reader.py`):

* the line is split into ink components, with dots merged into the letter body they
  belong to and a neighbouring mark (the colon of the label) dropped by ink mass,
* Persian is cursive, so a component is a *group* of one to four joined letters; the
  group is kept intact and read as a group instead of being cut at an arbitrary column,
* the pieces are read right to left and simply concatenated — that is the writing order,
* a bare vertical bar is decided by its shape (it is «ا»; this tesseract build answers
  «۱» or nothing at all for a two-pixel-wide stroke, and a name holds no digits),
* the crop keeps its black-on-white polarity and gets no white frame: both were measured
  to be the difference between «حسین» and an empty string.

### A bright or overexposed photo no longer moves the boxes

The design (smart card vs. the older front) is chosen without OCR, by measuring how
much of each candidate *ID-photo region* is ink. That measurement used to be a mean
grey level, and on a washed-out photo — highlights blown to white, contrast down to
40 — the two regions differed by 0.055, just under the 0.06 margin, so the pipeline
silently fell back to the default template. On a smart card the front template is one
row off: «نام» then reads the national code and «نام خانوادگی» reads the first name.

The score is now the *fraction* of pixels below the card's own ink level, which is
exposure independent: a portrait covers 15–47 % of its box, a column of text 2–9 %
(measured on three real photos against twelve benchmark renders, which it separates
2–8× apart). Three smaller fixes came with it: a name ROI may grow 9 % of the canvas to
the left when its first column still carries ink (a long «کمالی پور فهدان» was cut at
its «ن», while «نام پدر» sits to the right, so nothing else can be pulled in), a numeric
ROI grows 15 % the same way (a ten-digit national code was read as «168397» when its box
started inside the row), and a box that is still *cutting through glyphs* at its top or
bottom edge grows up to 3 % of the canvas until the glyphs end — measured on the edges
themselves, so a box with clear space around it never moves, and only on a card that was
*cut out of a larger picture* (a scan or a render warps exactly onto the template, and
growing there would pull the label into the value). On the real photos that last one is
worth more than everything else in this section — 1.9 % → 7.8 % of cards completely
right — and on the photo-style renders 81 % → 83 % of cards and 85.2 % → 89.1 % of
characters, with the clean renders unchanged at 95.2 %. The eyes that measured the row
are the same ones that read it — a washed-out, sideways photo of a smart card went from
*nothing* to national code, father name and birth date.

### Upload the card, not a screenshot of the result

The boxes this app draws can end up *inside* a later upload: if the picture you send is a
screenshot of the annotated preview, the rectangles are part of the pixels, and their
straight sides run across the very letters they point at. `strip_annotation_overlay`
removes them, but it cannot restore what they covered — the letters under an opaque box
are simply gone (on a 715-px screenshot «منصوری» loses its «ن» and «رضا» is cut in half).

The cleanup only fires when the picture really is an annotated screenshot: at least 0.4 %
of it must be saturated orange/green *and* a quarter of that colour must lie on long
straight lines. A real card photo scores 0.1 % and 0.0 there, and the warm tint of a
yellow-ish photo scores 25 % coverage but only 0.12 on the line test, so it is left alone
(every benchmark number in this file is unchanged by the cleanup).

The reconstruction competes with the whole-line readings in the same vote, and it can
only take over a *validated* whole-line reading when it is clearly more confident **and**
not shorter (a shorter reconstruction has dropped a component).

### Numeric fields are read glyph by glyph — on a flattened line

Tesseract's Persian model is weak on *runs* of digits (it prefers letter sequences), so
numeric fields are segmented by column projection and read one glyph at a time with a
digit whitelist. That only works when the segmentation is clean, and on a real card it
is not: the guilloche behind the digits survives binarisation and glues the row into a
single blob, and a line taller than ~60 px loses the digits at its ends. A ten-digit
national code was arriving as three glyphs.

``pipeline._glyph_line`` therefore flattens the ROI background and normalises it to the
same canonical height as every other field before segmentation.

A second real-card trap is the ``/`` of a date: it is a thin diagonal that the
classifier calls a ``۸``. The row then carries two glyphs too many, the groups are not
4/2/2 and the calendar validator never sees a date at all. ``digit_reader`` re-labels
the two least confident glyphs as separators, offers the resulting ``YYYY/MM/DD``
strings to the validator, and dates keep a second line height (1.6×) as a second
opinion — on 30 degraded photos the two heights read 18/30 and 19/30 birth dates
alone but 23/30 together, because they fail on different cards.

### Why line height is normalised

Tesseract's line finder has a narrow comfort zone. A text line whose height is below
~20 px loses its strokes, and above ~60 px the same line is silently dropped — ``psm 7``
returns an empty string and no error at all. That is exactly what happened to the field
ROIs of a real card: the photo was supersampled to a 1518×957 canvas, every name crop
came out ~80 px tall, and the reading collapsed into noise even though the crop was
perfectly legible.

Every ROI is therefore trimmed to its ink and scaled to a fixed line height
(``nidocr/vision/enhance.py``: ``fit_line``), and the vote runs at two heights (44 px
and 60 px) because blurred photos prefer the larger one. The same normalisation is what
makes a reading independent of the input resolution: a 470-px screenshot and a 1012-px
scan now produce the same OCR input.

One related trap: a white border around the line hurts rather than helps with this
Tesseract build (Persian text inside a white frame is dropped outright), so the fitted
line is passed through as-is.

### Two card designs

| template | design | fields |
|---|---|---|
| `national_card_front` | older plastic card (photo right) | شماره ملی، نام، نام خانوادگی، نام پدر، تاریخ تولد، شماره سریال، جنسیت |
| `national_card_smart` | modern «کارت هوشمند ملی» (photo left) | شماره ملی، نام، نام خانوادگی، تاریخ تولد، نام پدر، **پایان اعتبار** |

`template="auto"` picks the design from the position of the ID photo (no OCR, no extra
cost); an unknown template name falls back to the configured default instead of
raising, so a stale client can never break a page.

---

## How it works

```
image ─► card detection ─► perspective warp ─► background flattening ─► CLAHE/sharpening
                              │
                              ├─► label anchors (نام, کد ملی, …) ─┐
                              │                                   ├─► ROI per field
                              └─► calibrated template boxes ──────┘
                                            │
              ┌─────────────────────────────┴──────────────────────────────┐
              │ numeric fields                            text fields      │
              │ glyph segmentation                                        │
              │ (Sauvola + Otsu)                                          │
              │ → Persian-digit k-NN (6000 glyphs)                        │
              │ → vote with the known digit count                         │
              │ → alternatives (drop-one / runner-up / combined)          │
              │ → mod-11 checksum & Jalali calendar pick the value        │
              │                                      Tesseract psm 7 +    │
              │                                      sparse-text fallback │
              │                                      → name lexicon       │
              └─────────────────────────────┬──────────────────────────────┘
                                            ▼
                        fields + confidence + validation + bbox
```

Why this design (all measured during development):

* **Tesseract is the wrong tool for a *run* of Persian digits.** Its Persian model
  prefers letter sequences and drops/mutates digits: 20–50 % character accuracy on a
  10-digit national code. Segmenting the digits and classifying them one by one
  reaches **240/240 correct glyphs** on aligned cards.
* **The glyph count is a free prior.** A national code has 10 digits, a birth date
  8 + two `/`. Voting two independent binarisations (Sauvola, Otsu) against that
  count fixed most residual errors.
* **Validators are the oracle.** The mod-11 checksum always picks the right string
  when the alternatives contain it, so the reader offers alternatives instead of
  guessing; a single wrong digit is repaired *only* when the repair is unique
  (a blind "find any valid code" search always succeeds and would corrupt good data).
* **Persian names are a closed-ish vocabulary.** A name lexicon (17 500 first names,
  29 000 surnames, frequency-ordered) snaps OCR output that differs only by letter
  dots — `رضابی → رضایی`, `پیات → بیات`, `ارمین → آرمین` — and *never* adds or drops
  a letter, so it cannot invent a different name.
* **Free text has to keep its full width.** Column-projection refinement collapsed
  two-word names onto one letter; names are now refined row-wise only, which alone
  moved name accuracy by +19 points.

---

## Install

Requirements: Python ≥ 3.10, a Tesseract 5 binary, ~40 MB of disk for models.

```bash
git clone https://github.com/yazdanparasthesam/OCR-National-Card
cd OCR-National-Card
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# Tesseract 5 (any of these)
sudo apt install tesseract-ocr                   # Debian/Ubuntu
brew install tesseract                           # macOS
# ...or point NIDOCR_TESSERACT_CMD at your own build

python scripts/fetch_models.py                   # tessdata (fas, eng, osd) + checks
nidocr doctor                                    # verify everything
```

`nidocr doctor` prints the binary in use, the tessdata folder, the available
languages, the digit-classifier accuracy and the loaded lexicon — if the engine is
not ready it says exactly what is missing.

The repository already contains:

* `models/tessdata/` — `fas`, `eng`, `osd`
* `models/persian_digits.npz` — the Persian digit classifier (6 000 glyphs, k-NN)
* `data/lexicon/` — first names and surnames (see [`data/lexicon/README.md`](data/lexicon/README.md))
* `assets/fonts/` — Vazirmatn, used by the UI and the synthetic renderer

---

## Streamlit UI

```bash
nidocr ui                     # http://localhost:8501
# or: streamlit run nidocr/ui/app.py
```

* **Self-contained**: the font is embedded as base64 and the pipeline runs
  in-process, so it works inside an iframe / offline environment (no CDN, no
  localhost calls from the browser).
* **Behind a proxy**: `.streamlit/config.toml` turns Streamlit's XSRF check off and
  raises the upload limit to 50 MB. A reverse proxy that terminates TLS (the preview
  host, nginx, a tunnel) does not forward the session cookie the check wants, so every
  upload is answered with `403 XSRF token missing or invalid` — in the UI that shows up
  as *"AxiosError: Request failed with status code 403"* on the drop zone. Delete the
  file for a public, directly exposed deployment.
* Drag & drop, camera capture, or **«ساخت کارت نمونه»** which renders a synthetic
  card so you can try the whole flow without any data.
* Persian-first RTL layout: field cards with the value in Persian digits, a
  confidence bar, a validator badge («معتبر» / «اصلاح‌شده» / «بازبینی») and the raw
  OCR string underneath.
* Tabs: **JSON** (downloadable), **تحلیل اطمینان** (confidence bars + image-quality
  advice), **ناحیه‌های فیلد** (per-field ROI crops) and **دقت این کارت** — per-field +
  overall accuracy, estimated from the card itself or measured against values you type.
* **📊 سنجش دقت مدل** (expandable panel above the uploader): measure the model on
  N synthetic cards (`clean` / `photo` / `hard`, any seed) or on your own annotated
  folder, with a progress bar, per-field table, bar chart, misread list and
  Markdown/JSON download. The sidebar's «شرایط تصویر» controls the assumption used
  by the per-card estimate.
* The demo card survives reruns, so you can switch tabs and submit the accuracy form
  without losing the result.
* The card is shown with the detected boxes overlaid; the preprocessing result is
  available in an expander.

---

## FastAPI

```bash
nidocr serve --port 8000
# or: uvicorn nidocr.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs** · OpenAPI: `/openapi.json`

### One route per field

| method | path | what it does |
|---|---|---|
| `GET` | `/api/v1/health` | engine ready? languages, tessdata, templates, fields |
| `GET` | `/api/v1/capabilities` | lexicon sizes + feature flags |
| `POST` | `/api/v1/card` | **whole card** → every field |
| `POST` | `/api/v1/card/batch` | up to 12 cards in one request |
| `GET` | `/api/v1/card/templates` | template names |
| `GET` | `/api/v1/fields` | list of fields with Persian labels |
| `GET` | `/api/v1/fields/{field}` | metadata of one field |
| `POST` | `/api/v1/fields/{field}` | **read one field** (fast: only that ROI) |
| `GET` | `/api/v1/templates` | templates with canvas + field list |
| `GET` | `/api/v1/templates/{name}` | one template |
| `POST` | `/api/v1/validate/national-code` | mod-11 check without an image |
| `POST` | `/api/v1/validate/birth-date` | Jalali check without an image |

Field keys: `national_code`, `last_name`, `first_name`, `father_name`,
`birth_date`, `serial_number`, `gender`.

```bash
curl -F "file=@card.jpg" localhost:8000/api/v1/card | jq '.fields | map_values(.value)'
```

```json
{
  "national_code": "1344005985",
  "last_name": "بیات",
  "first_name": "نازنین",
  "father_name": "ابراهیم",
  "birth_date": "1340/05/28",
  "serial_number": "1132612",
  "gender": "زن"
}
```

```bash
# a single field (same pipeline, only the requested region is OCR'd)
curl -F "file=@card.jpg" localhost:8000/api/v1/fields/national_code
```

```json
{
  "success": true,
  "field": {
    "key": "national_code",
    "label_fa": "شماره ملی",
    "value": "1344005985",
    "value_fa": "۱۳۴۴۰۰۵۹۸۵",
    "confidence": 95.7,
    "validated": true,
    "repaired": false,
    "message": "رقم کنترل معتبر است",
    "bbox": {"x": 0.36, "y": 0.138, "width": 0.31, "height": 0.062, "space": "card"}
  },
  "processing_ms": 214.0
}
```

`POST /api/v1/card` accepts the extra form fields `template` (e.g.
`national_card_back`) and `debug=true` (adds the rectified + annotated images as
base64 JPEG and a per-field ROI). Errors follow a stable shape
`{"detail", "code", "extra"}` with 413 (too large), 415 (wrong type), 422 (unreadable
image / empty upload) and 503 (engine unavailable).

---

## CLI

```bash
nidocr read card.jpg                    # Persian table (digits in Persian)
nidocr read photos/ --json out.json     # a whole folder → one JSON
nidocr read card.jpg --latin-digits     # 1344005985 instead of ۱۳۴۴۰۰۵۹۸۵
nidocr field national_code card.jpg     # one field
nidocr fields                           # what can be read (labels + kinds)
nidocr check 0012345678                 # mod-11 checksum + repair
nidocr date 1360/01/01                  # Jalali validation
nidocr templates                        # template names
nidocr bench --generate 30 --difficulty photo
nidocr doctor                           # environment diagnostics
nidocr serve / nidocr ui                # the two servers
```

Exit code is `0` when the card was read (and the field validated), `1` otherwise —
handy in a batch script.

---

## Python API

```python
from nidocr.pipeline import get_ocr

ocr = get_ocr()                       # process-wide singleton
card = ocr.extract("card.jpg", debug=True)

card.flat()                           # {'national_code': '1344005985', ...}
card.get("birth_date")                # '1340/05/28'
card.fields["national_code"].value_fa # '۱۳۴۴۰۰۵۹۸۵'
card.fields["last_name"].validated    # True
card.ordered_fields()                 # in card reading order
card.overlay_png                      # base64 annotated image (debug=True)

single = ocr.extract_field("card.jpg", "national_code")
single.field.confidence, single.field.message
```

`extract` accepts a path, raw bytes, a file-like object, a PIL image or a numpy
array (BGR/gray). A `CardResult` is a pydantic model, so
`card.model_dump_json()` goes straight into your API or database.

Lower-level pieces are importable too: `nidocr.vision.card_detector.detect_card`,
`nidocr.ocr.digit_reader.read_digit_glyphs`, `nidocr.fields.validators.*`,
`nidocr.synthetic.renderer.render_card`.

---

## Templates & new card revisions

Field geometry lives in `nidocr/templates/*.json` (normalised 0..1 boxes for the
label and the value of every field, plus decoration boxes). Two templates ship:
`national_card_front` (7 fields) and `national_card_back` (national code, birth
date, serial).

The pipeline uses **both** the template and label anchors detected on the card, so it
tolerates the small print shifts between card revisions. If a new revision moves the
fields further, calibrate a template from a few sharp cards:

```bash
python scripts/calibrate_template.py cards/*.jpg --name my_revision --out nidocr/templates/my_revision.json
nidocr read card.jpg --template my_revision
```

The script rectifies the cards exactly like the pipeline, finds the printed labels,
takes the median box across the cards and writes the JSON — review it with
`--debug` before relying on it.

---

## Configuration

Every setting can be set as an environment variable with the `NIDOCR_` prefix (see
[`.env.example`](.env.example)):

| variable | default | meaning |
|---|---|---|
| `NIDOCR_TESSERACT_CMD` | auto-detected | path to the `tesseract` binary |
| `NIDOCR_TESSDATA_DIR` | `models/tessdata` | folder with `*.traineddata` |
| `NIDOCR_LANGUAGES` | `fas` | Tesseract languages |
| `NIDOCR_UPSCALE` | `3.0` | ROI upscale before OCR |
| `NIDOCR_PASSES` | `3` | preprocessing variants voted per field |
| `NIDOCR_USE_LABEL_ANCHORS` | `true` | detect labels to refine ROIs |
| `NIDOCR_USE_NAME_LEXICON` | `true` | snap names to the lexicon |
| `NIDOCR_LEXICON_DIR` | `data/lexicon` | replace with your own name lists |
| `NIDOCR_DEFAULT_TEMPLATE` | `national_card_front` | template used when none is given |
| `NIDOCR_MAX_UPLOAD_MB` | `12` | API upload limit |
| `NIDOCR_MAX_IMAGE_SIDE` | `2400` | images are downscaled to this side |
| `NIDOCR_BIRTH_YEAR_MIN/MAX` | `1200/1450` | sane Jalali birth years |
| `NIDOCR_API_PREFIX` | `/api/v1` | route prefix |
| `NIDOCR_CACHE_OCR` | `true` | cache OCR results per image |

---

## Project layout

```
nidocr/
  api/            FastAPI app + one router per resource (health, card, fields, templates, validate)
  ui/app.py       self-contained RTL Streamlit UI
  vision/         card_detector, enhance (flatten/CLAHE/sharpen), binarize (Sauvola/Otsu),
                  layout (templates, ROI refinement), orient (OSD rotation), annotate
  ocr/            tesseract_engine (subprocess + cache), digit_reader (glyph segmentation),
                  digit_classifier (k-NN), label_anchor, text (Persian normalisation)
  fields/         definitions (7 fields), validators (mod-11, Jalali, names), lexicon
  synthetic/      card renderer + random Persian identities (used for training/benchmarks)
  pipeline.py     the orchestrator: prepare → detect → read → validate → rank
  accuracy.py     measured (per field + overall) and estimated accuracy, shared by UI/CLI/bench
  cli.py, schemas.py, config.py, metrics.py, errors.py, image_io.py
  templates/      national_card_front.json, national_card_smart.json, national_card_back.json
data/lexicon/     first_names.txt, last_names.txt
models/           tessdata/ + persian_digits.npz
scripts/          benchmark, generate_synthetic_cards, train_digit_model,
                  build_name_lexicon, calibrate_template, fetch_models
tests/            114 tests (unit + API + end-to-end on a rendered card + accuracy)
reports/          measured benchmark + digit-model reports
```

---

## Tests

```bash
pytest                     # 114 tests, ~49 s
ruff check .               # style (optional)
```

The suite covers Persian normalisation, every validator (including the *unique-repair*
rule and Jalali edge cases), the lexicon's precision guarantees, card detection and
warping, ROI refinement, the digit reader, the whole pipeline on a rendered card
(`≥ 6/7` fields exact), the accuracy layer (measured per field + overall, the estimate's
bounds and its difficulty ordering) and the HTTP contract of the API (routes, errors,
OpenAPI).

Every module, class and function — 409 of them, tests and scripts included — carries a
docstring; `python -m pytest` and `ruff check .` are both clean.

---

## Limitations & roadmap

What the current version does **not** do, and where the next accuracy gains are:

1. **Small, blurred digits** (photographed cards: birth date 56.7 %, serial 53.3 %).
   The classifier's top-2 often does not contain the truth, so the validators have
   nothing to choose from. Next step: a small CNN on the same glyph features, and a
   super-resolution pass for cards below ~700 px wide.
2. **Date structure**: the reader now knows a date is `4/2/2` and drops surplus
   glyphs, but a misread digit inside a *valid* date cannot be detected — the
   calendar is a weak oracle there. Month/day priors (or a second independent read)
   would help.
3. **Misspellings in the public first-name list** (e.g. `خسین`) are treated as real
   names, so a dotted-letter error can survive. Feeding a civil-registry list via
   `NIDOCR_LEXICON_DIR` fixes this for production.
4. **Multi-digit serials with no checksum**: an extra segmented glyph is
   indistinguishable from a real digit without a length prior. Iranian serials are
   7–10 digits; a per-card-template length constraint would tighten this.
5. Back-of-card template is defined but only lightly benchmarked; rotated 180° cards
   are handled by the warp, not by content detection.
6. No handwriting, no liveness/hologram check, no OCR of the barcode (deliberately
   out of scope).

---

## Data sources & licence

* **Code**: MIT — see [`LICENSE`](LICENSE).
* **Persian name lexicon** (`data/lexicon/`): derived from
  [`nikahd99/iranian-Names-Database-By-Gender`](https://github.com/nikahd99/iranian-Names-Database-By-Gender)
  and [`farbodbj/iranian-surname-frequencies`](https://github.com/farbodbj/iranian-surname-frequencies)
  (Apache-2.0); only normalised unique names are stored. Rebuild with
  `python scripts/build_name_lexicon.py` — see [`data/lexicon/README.md`](data/lexicon/README.md).
* **Tesseract models**: `tessdata_fast` (Apache-2.0), fetched by `scripts/fetch_models.py`.
* **Fonts**: [Vazirmatn](https://github.com/rastikerdar/vazirmatn) (SIL OFL 1.1);
  the training fonts under `assets/fonts/train/` are the same family plus Sahel,
  Samim and Shabnam (SIL OFL 1.1).
* The digit classifier, the templates and the synthetic card renderer are original
  work included in this repository.

> The synthetic cards are generated from random identities; no real personal data is
> included in this repository. If you process real cards, you are responsible for the
> applicable data-protection rules (e.g. Iran's personal data regulations and the
> GDPR for EU residents) — the API and UI do not store images or results anywhere.
