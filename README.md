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

Field **exact match** / **character accuracy** — 30 cards each, 7 fields per card,
Tesseract 5.4.1 (`fas.traineddata`, `tessdata_fast`) on a 4-core CPU:

| field | clean (exact) | clean (chars) | photo (exact) | photo (chars) |
|---|---|---|---|---|
| `national_code` | **100.0 %** | 100.0 % | 90.0 % | 92.7 % |
| `last_name` | **96.7 %** | 98.0 % | 90.0 % | 91.8 % |
| `first_name` | 90.0 % | 94.2 % | 80.0 % | 87.5 % |
| `father_name` | 90.0 % | 97.8 % | 76.7 % | 82.7 % |
| `birth_date` | 80.0 % | 96.3 % | 56.7 % | 86.3 % |
| `serial_number` | 93.3 % | 98.6 % | 53.3 % | 73.9 % |
| `gender` | **100.0 %** | 100.0 % | 80.0 % | 80.0 % |
| **overall** | **92.9 %** | **97.8 %** | **75.2 %** | **85.0 %** |

Full reports: [`reports/benchmark_clean.md`](reports/benchmark_clean.md),
[`reports/benchmark_photo.md`](reports/benchmark_photo.md),
digit-classifier report [`reports/digits.md`](reports/digits.md).

How to read it:

* **character accuracy ≥ 97.8 % on flat scans**, i.e. ≤ 2 characters wrong per card —
  the values are safe to *review*, and the validator flags what it doubts.
* **national code and gender are effectively exact**: the mod-11 checksum and the
  two-word vocabulary make them self-correcting.
* **photographed cards are harder** (down to 0.62× resolution, blur σ≈0.9, JPEG 45–70
  and glare in the benchmark): 75 % of cards come out perfect, 85 % of characters
  right. Low-res dates and serials are where it hurts — see
  [roadmap](#limitations--roadmap).
* Latency: **1.1–1.5 s per card** on CPU (single Tesseract binary, 8–14 subprocess
  calls per card; the UI and API reuse a warm process-wide pipeline).

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

Nothing is inflated: the measured panel reports 92.9 % / 97.8 % on clean renders and
75.2 % / 85.0 % on photo-style cards — the numbers in the table above.

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
                  layout (templates, ROI refinement), annotate
  ocr/            tesseract_engine (subprocess + cache), digit_reader (glyph segmentation),
                  digit_classifier (k-NN), label_anchor, text (Persian normalisation)
  fields/         definitions (7 fields), validators (mod-11, Jalali, names), lexicon
  synthetic/      card renderer + random Persian identities (used for training/benchmarks)
  pipeline.py     the orchestrator: prepare → detect → read → validate → rank
  accuracy.py     measured (per field + overall) and estimated accuracy, shared by UI/CLI/bench
  cli.py, schemas.py, config.py, metrics.py, errors.py, image_io.py
  templates/      national_card_front.json, national_card_back.json
data/lexicon/     first_names.txt, last_names.txt
models/           tessdata/ + persian_digits.npz
scripts/          benchmark, generate_synthetic_cards, train_digit_model,
                  build_name_lexicon, calibrate_template, fetch_models
tests/            85 tests (unit + API + end-to-end on a rendered card + accuracy)
reports/          measured benchmark + digit-model reports
```

---

## Tests

```bash
pytest                     # 85 tests, ~21 s
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
