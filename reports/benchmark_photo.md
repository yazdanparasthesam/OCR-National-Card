# National card OCR benchmark

- dataset: `/tmp/final_photo`
- cards: **30**
- engine: tesseract (`fas`) + glyph digit classifier
- field exact match: **75.24%**
- field character accuracy: **84.97%**
- fully correct cards: **75.24%**
- mean latency: 1493 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 90.0% | 92.7% | 92.4 | 90.0% |
| `last_name` | 90.0% | 91.8% | 79.0 | 100.0% |
| `first_name` | 80.0% | 87.5% | 76.7 | 100.0% |
| `father_name` | 76.7% | 82.7% | 72.3 | 100.0% |
| `birth_date` | 56.7% | 86.3% | 70.1 | 80.0% |
| `serial_number` | 53.3% | 73.9% | 87.4 | 100.0% |
| `gender` | 80.0% | 80.0% | 78.4 | 83.3% |
