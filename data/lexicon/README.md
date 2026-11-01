# Persian name lexicon

Two plain-text files, one name per line, `UTF-8`, no header:

| file | entries | ordering | used for |
|---|---|---|---|
| `first_names.txt` | ~17 500 | source order (deduplicated) | `first_name`, `father_name` |
| `last_names.txt` | ~29 000 | **frequency** (most common first) | `last_name` |

The pipeline uses them to *snap* OCR output to real names. The rules are deliberately
precision-first:

* a snap is only attempted when the raw text is **not** already a lexicon hit;
* the candidate must have the **same length** — never add or drop a letter;
* every differing character must be a known Tesseract confusion
  (`ب ت ث ن پ ی ئ`, `س ش`, `ص ض`, `ط ظ`, `ق غ`, `ک گ`, `ه ة`, `ا آ أ إ`, `د ذ`, `ز ژ`, `ح خ ج`, `چ ج ح`);
* at most one substitution for names shorter than 6 letters, two for longer ones;
* `آ/أ/إ/ا`, `ة/ه`, `ؤ/و`, `ئ/ی` are folded when comparing, so `ارمین` matches `آرمین`
  and the lexicon's spelling is returned;
* for surnames (frequency-ordered) a later candidate must be clearly better
  (> 0.02 similarity) before it replaces a more common one.

Examples:

```
رضابی → رضایی    پیات → بیات    ارمین → آرمین    پارسابی → پارسایی
محمد  → محمد     (a dropped letter is never invented)
کامران → کامران  (out-of-lexicon names are kept as-is, just flagged)
```

## Sources

* **First names** — [nikahd99/iranian-Names-Database-By-Gender](https://github.com/nikahd99/iranian-Names-Database-By-Gender)
  (~20 k Iranian names tagged by gender, Apache-2.0).
* **Surnames** — [farbodbj/iranian-surname-frequencies](https://github.com/farbodbj/iranian-surname-frequencies)
  (100 k+ surnames with frequencies from ~10 M records, Apache-2.0). The repository
  stores the 30 000 most frequent.

Neither dataset is redistributed here: only the normalised, de-duplicated name
strings are. `scripts/build_name_lexicon.py` documents the exact transformation and
can rebuild both files from the upstream CSVs.

The public lists are crowd-sourced and contain some misspellings, which is why a
dotted-letter error can occasionally survive (`خسین` is in the list). For production,
point `NIDOCR_LEXICON_DIR` at a civil-registry list with the same file format; the
matcher and every other part of the pipeline stay unchanged.

## Using your own list

```bash
export NIDOCR_LEXICON_DIR=/path/to/my/lexicon   # must contain first_names.txt / last_names.txt
nidocr doctor | grep lexicon
```

Only lower-case Persian letters, spaces and ZWNJ are expected; anything else is
filtered out on load, and a missing file simply disables snapping for that kind.
