# National card OCR benchmark — clean scans

- dataset: `final_clean`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `clean`
- field exact match: **92.38%**
- field character accuracy: **97.68%**
- fully correct cards: **92.38%**
- mean latency: 1281 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 100.0% | 100.0% | 97.5 | 100.0% |
| `last_name` | 93.3% | 96.9% | 80.8 | 100.0% |
| `first_name` | 90.0% | 94.2% | 78.5 | 100.0% |
| `father_name` | 90.0% | 97.8% | 82.8 | 100.0% |
| `birth_date` | 80.0% | 96.3% | 89.0 | 93.3% |
| `serial_number` | 93.3% | 98.6% | 99.6 | 100.0% |
| `gender` | 100.0% | 100.0% | 92.2 | 100.0% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_005.jpg | `birth_date` | 1368/08/10 | 11568/08/10 | 95 |
| card_007.jpg | `father_name` | یدالله | بدالله | 73 |
| card_009.jpg | `serial_number` | 1815049 | 1875049 | 100 |
| card_010.jpg | `father_name` | حسین | خسین | 67 |
| card_014.jpg | `birth_date` | 1350/11/20 | 1350/01/20 | 43 |
| card_015.jpg | `serial_number` | 1103185 | 11015185 | 98 |
| card_019.jpg | `father_name` | اصغر | اصعر | 86 |
| card_020.jpg | `first_name` | یاسر | پاسر | 33 |
| card_021.jpg | `first_name` | شیدا | شبد | 58 |
| card_021.jpg | `birth_date` | 1331/11/11 | 1331/01/01 | 54 |
| card_023.jpg | `birth_date` | 1341/11/04 | 1349/11/04 | 95 |
| card_025.jpg | `last_name` | شیرازی | شیرازی و | 68 |
| card_025.jpg | `first_name` | هستی | پدرام | 89 |
| card_026.jpg | `birth_date` | 1351/10/30 | 1151/10/150 | 94 |
| card_027.jpg | `last_name` | شریفی | شریفی اه | 44 |
| card_029.jpg | `birth_date` | 1366/02/11 | 1366/02/08 | 29 |
