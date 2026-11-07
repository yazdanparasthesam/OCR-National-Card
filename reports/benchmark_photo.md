# National card OCR benchmark — phone photos

- dataset: `final_photo`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `photo`
- field exact match: **83.33%**
- field character accuracy: **89.11%**
- fully correct cards: **83.33%**
- mean latency: 1751 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 83.3% | 89.7% | 93.2 | 86.7% |
| `last_name` | 90.0% | 92.6% | 77.1 | 100.0% |
| `first_name` | 93.3% | 94.2% | 82.8 | 100.0% |
| `father_name` | 93.3% | 96.2% | 80.2 | 100.0% |
| `birth_date` | 66.7% | 86.7% | 81.6 | 86.7% |
| `serial_number` | 76.7% | 84.5% | 93.5 | 86.7% |
| `gender` | 80.0% | 80.0% | 75.6 | 80.0% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_001.jpg | `national_code` | 3421027978 | 13 | 86 |
| card_001.jpg | `first_name` | حمید | نام | 76 |
| card_001.jpg | `birth_date` | 1380/11/28 | 1338/01/08 | 31 |
| card_004.jpg | `last_name` | بیات | تاب | 29 |
| card_005.jpg | `national_code` | 7917122603 | 7917133605 | 47 |
| card_005.jpg | `last_name` | سلطانی | ای | 32 |
| card_005.jpg | `birth_date` | 1359/03/13 | 1301/01/53 | 59 |
| card_005.jpg | `serial_number` | 595664740 | 1 | 61 |
| card_005.jpg | `gender` | زن | — | 0 |
| card_006.jpg | `birth_date` | 1365/04/27 | 1337/04/08 | 35 |
| card_006.jpg | `serial_number` | 842172249 | 84217 | 97 |
| card_006.jpg | `gender` | مرد | — | 0 |
| card_007.jpg | `birth_date` | 1341/10/26 | 1347/10/26 | 97 |
| card_010.jpg | `national_code` | 4569488803 | 03 | 86 |
| card_010.jpg | `birth_date` | 1368/06/18 | 1368/08/00 | 35 |
| card_010.jpg | `serial_number` | 979012143 | 0144 | 78 |
| card_010.jpg | `gender` | زن | 0 | 44 |
| card_011.jpg | `national_code` | 0918202655 | 09186543 | 87 |
| card_011.jpg | `father_name` | ابراهیم | اباهیم | 48 |
| card_011.jpg | `birth_date` | 1363/08/14 | 1327/08/14 | 96 |
| card_011.jpg | `serial_number` | 89207094 | 9207094 | 58 |
| card_012.jpg | `last_name` | کاظمی | کی آ | 15 |
| card_012.jpg | `birth_date` | 1374/11/08 | 1318/56/66 | 79 |
| card_012.jpg | `serial_number` | 4841507 | 48357 | 64 |
| card_012.jpg | `gender` | زن | 0 | 59 |
| card_013.jpg | `first_name` | نیما | امد | 61 |
| card_013.jpg | `birth_date` | 1375/04/19 | 1301/01/43 | 57 |
| card_013.jpg | `serial_number` | 653898989 | 0 | 86 |
| card_013.jpg | `gender` | مرد | — | 0 |
| card_024.jpg | `birth_date` | 1377/07/12 | 1308/02/08 | 0 |
| card_028.jpg | `national_code` | 1034052314 | 1034 | 93 |
| card_028.jpg | `father_name` | حسین | تاریخ تولد | 73 |
| card_028.jpg | `birth_date` | 1352/10/27 | 1301/01/31 | 56 |
| card_028.jpg | `serial_number` | 92319173 | 0 | 86 |
| card_028.jpg | `gender` | مرد | — | 0 |
