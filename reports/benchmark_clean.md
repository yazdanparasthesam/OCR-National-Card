# National card OCR benchmark — clean scans

- dataset: `final_clean`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `clean`
- field exact match: **94.76%**
- field character accuracy: **98.81%**
- fully correct cards: **94.76%**
- mean latency: 2150 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 100.0% | 100.0% | 98.3 | 100.0% |
| `last_name` | 100.0% | 100.0% | 88.4 | 100.0% |
| `first_name` | 93.3% | 97.5% | 87.0 | 100.0% |
| `father_name` | 90.0% | 97.8% | 86.6 | 100.0% |
| `birth_date` | 83.3% | 97.3% | 92.1 | 96.7% |
| `serial_number` | 96.7% | 99.0% | 99.2 | 100.0% |
| `gender` | 100.0% | 100.0% | 91.7 | 100.0% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_007.jpg | `father_name` | یدالله | پدالله | 75 |
| card_008.jpg | `birth_date` | 1346/07/11 | 1346/07/01 | 96 |
| card_010.jpg | `father_name` | حسین | خسین | 82 |
| card_012.jpg | `first_name` | حسین | اخسین | 83 |
| card_014.jpg | `birth_date` | 1350/11/20 | 11550/01/20 | 94 |
| card_015.jpg | `serial_number` | 1103185 | 11015185 | 94 |
| card_017.jpg | `birth_date` | 1383/12/19 | 1383/02/19 | 97 |
| card_018.jpg | `father_name` | حسین | خسین | 77 |
| card_020.jpg | `first_name` | یاسر | پاسر | 85 |
| card_020.jpg | `birth_date` | 1369/12/12 | 1369/02/12 | 96 |
| card_021.jpg | `birth_date` | 1331/11/11 | 1331/01/01 | 20 |
