# National card OCR benchmark — clean scans

- dataset: `final_clean`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `clean`
- field exact match: **94.76%**
- field character accuracy: **98.93%**
- fully correct cards: **94.76%**
- mean latency: 1641 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 100.0% | 100.0% | 97.5 | 100.0% |
| `last_name` | 100.0% | 100.0% | 88.3 | 100.0% |
| `first_name` | 93.3% | 98.3% | 85.3 | 100.0% |
| `father_name` | 93.3% | 98.6% | 85.7 | 100.0% |
| `birth_date` | 83.3% | 97.0% | 86.0 | 93.3% |
| `serial_number` | 93.3% | 98.6% | 99.6 | 100.0% |
| `gender` | 100.0% | 100.0% | 91.7 | 100.0% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_005.jpg | `birth_date` | 1368/08/10 | 11568/08/10 | 95 |
| card_007.jpg | `father_name` | یدالله | پدالله | 75 |
| card_009.jpg | `serial_number` | 1815049 | 1875049 | 100 |
| card_012.jpg | `first_name` | حسین | حخسین | 75 |
| card_014.jpg | `birth_date` | 1350/11/20 | 11550/11/20 | 95 |
| card_015.jpg | `serial_number` | 1103185 | 11015185 | 98 |
| card_018.jpg | `father_name` | حسین | خسین | 58 |
| card_020.jpg | `first_name` | یاسر | پاسر | 85 |
| card_021.jpg | `birth_date` | 1331/11/11 | 1331/01/01 | 20 |
| card_023.jpg | `birth_date` | 1341/11/04 | 1349/11/04 | 95 |
| card_029.jpg | `birth_date` | 1366/02/11 | 1366/03/10 | 8 |
