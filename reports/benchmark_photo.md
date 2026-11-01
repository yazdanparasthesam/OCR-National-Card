# National card OCR benchmark — phone photos

- dataset: `final_photo`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `photo`
- field exact match: **79.52%**
- field character accuracy: **84.22%**
- fully correct cards: **79.52%**
- mean latency: 2024 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 90.0% | 94.0% | 93.1 | 93.3% |
| `last_name` | 80.0% | 82.0% | 78.4 | 100.0% |
| `first_name` | 83.3% | 88.3% | 81.0 | 100.0% |
| `father_name` | 86.7% | 86.7% | 82.5 | 100.0% |
| `birth_date` | 73.3% | 87.7% | 81.7 | 86.7% |
| `serial_number` | 66.7% | 74.2% | 88.5 | 86.7% |
| `gender` | 76.7% | 76.7% | 76.6 | 76.7% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_001.jpg | `national_code` | 3421027978 | 3431037976 | 56 |
| card_001.jpg | `last_name` | نیکخواه | کر | 32 |
| card_004.jpg | `last_name` | بیات | تاب | 29 |
| card_005.jpg | `father_name` | محمدرضا | تاریخ تولد | 66 |
| card_005.jpg | `birth_date` | 1359/03/13 | 0781/08/30 | 77 |
| card_005.jpg | `serial_number` | 595664740 | 1 | 62 |
| card_005.jpg | `gender` | زن | — | 0 |
| card_006.jpg | `father_name` | منصور | تاررخ تولد | 73 |
| card_006.jpg | `birth_date` | 1365/04/27 | 1377/25/27 | 0 |
| card_006.jpg | `serial_number` | 842172249 | 912206 | 13 |
| card_006.jpg | `gender` | مرد | — | 0 |
| card_009.jpg | `serial_number` | 23645293 | 24442 | 84 |
| card_010.jpg | `national_code` | 4569488803 | 03 | 86 |
| card_010.jpg | `last_name` | گودرزی | نام خانوادگی | 89 |
| card_010.jpg | `first_name` | زهرا | رهرا | 74 |
| card_010.jpg | `birth_date` | 1368/06/18 | 1301/06/08 | 0 |
| card_010.jpg | `serial_number` | 979012143 | 0144 | 86 |
| card_010.jpg | `gender` | زن | 7 | 58 |
| card_011.jpg | `national_code` | 0918202655 | 96861 | 89 |
| card_011.jpg | `last_name` | رضایی | آور | 64 |
| card_011.jpg | `birth_date` | 1363/08/14 | 1327/08/14 | 98 |
| card_011.jpg | `gender` | زن | مت | 63 |
| card_012.jpg | `last_name` | کاظمی | کر | 9 |
| card_012.jpg | `first_name` | فرشته | پدر | 71 |
| card_012.jpg | `birth_date` | 1374/11/08 | 1115/55/43 | 81 |
| card_012.jpg | `serial_number` | 4841507 | 101337507 | 33 |
| card_012.jpg | `gender` | زن | 9 ۰ | 77 |
| card_013.jpg | `first_name` | نیما | امده | 35 |
| card_013.jpg | `father_name` | جواد | تاریخ تولد | 77 |
| card_013.jpg | `birth_date` | 1375/04/19 | 1379/04/09 | 45 |
| card_013.jpg | `serial_number` | 653898989 | 0 | 72 |
| card_013.jpg | `gender` | مرد | — | 0 |
| card_014.jpg | `first_name` | آیدا | ایدا | 83 |
| card_016.jpg | `birth_date` | 1352/10/15 | 1321/01/15 | 79 |
| card_016.jpg | `serial_number` | 4435624 | 44324 | 81 |
| card_023.jpg | `serial_number` | 41376775 | 84723 | 73 |
| card_026.jpg | `serial_number` | 03996580 | 34993 | 70 |
| card_028.jpg | `last_name` | شریفی | تام | 66 |
| card_028.jpg | `first_name` | فرشید | پدر | 92 |
| card_028.jpg | `father_name` | حسین | تاریخ تولد | 73 |
