# National card OCR benchmark

- dataset: `/tmp/final_clean`
- cards: **30**
- engine: tesseract (`fas`) + glyph digit classifier
- field exact match: **92.86%**
- field character accuracy: **97.84%**
- fully correct cards: **92.86%**
- mean latency: 1190 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 100.0% | 100.0% | 97.5 | 100.0% |
| `last_name` | 96.7% | 98.0% | 80.8 | 100.0% |
| `first_name` | 90.0% | 94.2% | 78.5 | 100.0% |
| `father_name` | 90.0% | 97.8% | 82.8 | 100.0% |
| `birth_date` | 80.0% | 96.3% | 89.0 | 93.3% |
| `serial_number` | 93.3% | 98.6% | 99.6 | 100.0% |
| `gender` | 100.0% | 100.0% | 92.2 | 100.0% |
