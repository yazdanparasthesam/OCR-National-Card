# National card OCR benchmark — phone photos

- dataset: `final_photo`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `photo`
- field exact match: **79.52%**
- field character accuracy: **84.71%**
- fully correct cards: **79.52%**
- mean latency: 1671 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 86.7% | 91.0% | 96.8 | 86.7% |
| `last_name` | 86.7% | 88.5% | 76.7 | 100.0% |
| `first_name` | 86.7% | 92.3% | 75.5 | 100.0% |
| `father_name` | 80.0% | 83.6% | 73.2 | 100.0% |
| `birth_date` | 73.3% | 85.3% | 86.5 | 83.3% |
| `serial_number` | 66.7% | 75.6% | 87.6 | 90.0% |
| `gender` | 76.7% | 76.7% | 74.0 | 80.0% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_001.jpg | `national_code` | 3421027978 | 3421616 | 88 |
| card_001.jpg | `last_name` | نیکخواه | تام | 79 |
| card_004.jpg | `last_name` | بیات | تب | 66 |
| card_005.jpg | `father_name` | محمدرضا | تاریخ تولد | 67 |
| card_005.jpg | `birth_date` | 1359/03/13 | 0781/08/30 | 77 |
| card_005.jpg | `serial_number` | 595664740 | 1 | 62 |
| card_005.jpg | `gender` | زن | — | 0 |
| card_006.jpg | `birth_date` | 1365/04/27 | 1225/07/23 | 0 |
| card_006.jpg | `serial_number` | 842172249 | 42123 | 0 |
| card_006.jpg | `gender` | مرد | — | 0 |
| card_009.jpg | `serial_number` | 23645293 | 24442 | 84 |
| card_010.jpg | `national_code` | 4569488803 | 03 | 86 |
| card_010.jpg | `last_name` | گودرزی | نام خانوادگی | 82 |
| card_010.jpg | `first_name` | زهرا | هرا | 86 |
| card_010.jpg | `serial_number` | 979012143 | 0135133 | 56 |
| card_011.jpg | `national_code` | 0918202655 | 96861 | 89 |
| card_011.jpg | `last_name` | رضایی | دضاد | 66 |
| card_011.jpg | `birth_date` | 1363/08/14 | 1327/08/14 | 98 |
| card_011.jpg | `gender` | زن | مرد | 21 |
| card_012.jpg | `first_name` | فرشته | رمان | 34 |
| card_012.jpg | `father_name` | علی | اسنا | 36 |
| card_012.jpg | `birth_date` | 1374/11/08 | 1115/55/43 | 81 |
| card_012.jpg | `serial_number` | 4841507 | 13457 | 48 |
| card_012.jpg | `gender` | زن | ۰ | 63 |
| card_013.jpg | `father_name` | جواد | تاریخ تولد | 78 |
| card_013.jpg | `birth_date` | 1375/04/19 | — | 83 |
| card_013.jpg | `serial_number` | 653898989 | 0 | 72 |
| card_013.jpg | `gender` | مرد | ۰۲ ۲۰ 1 | 61 |
| card_014.jpg | `first_name` | آیدا | ایدا | 81 |
| card_016.jpg | `birth_date` | 1352/10/15 | 1321/01/15 | 79 |
| card_016.jpg | `serial_number` | 4435624 | 44324 | 81 |
| card_016.jpg | `gender` | زن | 3 (1۱۳ | 48 |
| card_023.jpg | `birth_date` | 1370/01/05 | 11570/01/05 | 94 |
| card_023.jpg | `serial_number` | 41376775 | 84723 | 73 |
| card_025.jpg | `father_name` | یدالله | یداله | 68 |
| card_026.jpg | `serial_number` | 03996580 | 34993 | 70 |
| card_028.jpg | `national_code` | 1034052314 | 130340 | 81 |
| card_028.jpg | `first_name` | فرشید | بسم | 59 |
| card_028.jpg | `father_name` | حسین | ااآ | 20 |
| card_028.jpg | `birth_date` | 1352/10/27 | 5781/08/50 | 75 |
