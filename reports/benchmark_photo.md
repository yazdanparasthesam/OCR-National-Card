# National card OCR benchmark — phone photos

- dataset: `final_photo`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `photo`
- field exact match: **80.48%**
- field character accuracy: **85.00%**
- fully correct cards: **80.48%**
- mean latency: 1847 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 90.0% | 94.7% | 92.4 | 93.3% |
| `last_name` | 80.0% | 82.0% | 78.4 | 100.0% |
| `first_name` | 83.3% | 88.3% | 81.0 | 100.0% |
| `father_name` | 86.7% | 86.7% | 82.5 | 100.0% |
| `birth_date` | 73.3% | 88.0% | 85.3 | 86.7% |
| `serial_number` | 73.3% | 78.7% | 89.9 | 86.7% |
| `gender` | 76.7% | 76.7% | 76.6 | 76.7% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_001.jpg | `national_code` | 3421027978 | 3431037976 | 56 |
| card_001.jpg | `last_name` | نیکخواه | کر | 32 |
| card_004.jpg | `last_name` | بیات | تاب | 29 |
| card_005.jpg | `father_name` | محمدرضا | تاریخ تولد | 66 |
| card_005.jpg | `birth_date` | 1359/03/13 | — | 72 |
| card_005.jpg | `serial_number` | 595664740 | 1 | 61 |
| card_005.jpg | `gender` | زن | — | 0 |
| card_006.jpg | `father_name` | منصور | تاررخ تولد | 73 |
| card_006.jpg | `birth_date` | 1365/04/27 | 1377/25/27 | 0 |
| card_006.jpg | `serial_number` | 842172249 | 912206 | 13 |
| card_006.jpg | `gender` | مرد | — | 0 |
| card_007.jpg | `birth_date` | 1341/10/26 | 1347/10/26 | 97 |
| card_009.jpg | `serial_number` | 23645293 | 23634 | 82 |
| card_010.jpg | `national_code` | 4569488803 | 03 | 86 |
| card_010.jpg | `last_name` | گودرزی | نام خانوادگی | 89 |
| card_010.jpg | `first_name` | زهرا | رهرا | 74 |
| card_010.jpg | `birth_date` | 1368/06/18 | 1301/06/08 | 0 |
| card_010.jpg | `serial_number` | 979012143 | 0144 | 78 |
| card_010.jpg | `gender` | زن | 7 | 58 |
| card_011.jpg | `national_code` | 0918202655 | 09186543 | 87 |
| card_011.jpg | `last_name` | رضایی | آور | 64 |
| card_011.jpg | `birth_date` | 1363/08/14 | 1327/08/14 | 96 |
| card_011.jpg | `gender` | زن | مت | 63 |
| card_012.jpg | `last_name` | کاظمی | کر | 9 |
| card_012.jpg | `first_name` | فرشته | پدر | 71 |
| card_012.jpg | `birth_date` | 1374/11/08 | 1311/85/55 | 73 |
| card_012.jpg | `serial_number` | 4841507 | 101337507 | 33 |
| card_012.jpg | `gender` | زن | 9 ۰ | 77 |
| card_013.jpg | `first_name` | نیما | امده | 35 |
| card_013.jpg | `father_name` | جواد | تاریخ تولد | 77 |
| card_013.jpg | `birth_date` | 1375/04/19 | 1379/04/09 | 45 |
| card_013.jpg | `serial_number` | 653898989 | 0 | 86 |
| card_013.jpg | `gender` | مرد | — | 0 |
| card_014.jpg | `first_name` | آیدا | ایدا | 83 |
| card_016.jpg | `serial_number` | 4435624 | 44371 | 81 |
| card_028.jpg | `last_name` | شریفی | تام | 66 |
| card_028.jpg | `first_name` | فرشید | پدر | 92 |
| card_028.jpg | `father_name` | حسین | تاریخ تولد | 73 |
| card_028.jpg | `birth_date` | 1352/10/27 | 0781/08/35 | 78 |
| card_028.jpg | `serial_number` | 92319173 | 0 | 86 |
