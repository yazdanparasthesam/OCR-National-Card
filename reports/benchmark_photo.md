# National card OCR benchmark — phone photos

- dataset: `final_photo`
- engine: tesseract (`fas`) + glyph digit classifier

- cards: **30**
- difficulty: `photo`
- field exact match: **75.24%**
- field character accuracy: **84.97%**
- fully correct cards: **75.24%**
- mean latency: 1470 ms / card

| field | exact match | char accuracy | mean confidence | validator pass |
|---|---|---|---|---|
| `national_code` | 90.0% | 92.7% | 92.4 | 90.0% |
| `last_name` | 90.0% | 91.8% | 79.0 | 100.0% |
| `first_name` | 80.0% | 87.5% | 76.7 | 100.0% |
| `father_name` | 76.7% | 82.7% | 72.3 | 100.0% |
| `birth_date` | 56.7% | 86.3% | 70.1 | 80.0% |
| `serial_number` | 53.3% | 73.9% | 87.4 | 100.0% |
| `gender` | 80.0% | 80.0% | 78.4 | 83.3% |

### Misreads

| image | field | expected | predicted | conf |
|---|---|---|---|---|
| card_001.jpg | `national_code` | 3421027978 | 346111 | 88 |
| card_001.jpg | `last_name` | نیکخواه | تام | 79 |
| card_001.jpg | `birth_date` | 1380/11/28 | 1380/17/28 | 97 |
| card_005.jpg | `first_name` | نرگس | ترگس | 68 |
| card_005.jpg | `father_name` | محمدرضا | محمد زضا | 78 |
| card_005.jpg | `birth_date` | 1359/03/13 | 1359/83/134 | 7 |
| card_005.jpg | `serial_number` | 595664740 | 59566 | 100 |
| card_005.jpg | `gender` | زن | ات۱۳۲۳ | 40 |
| card_006.jpg | `father_name` | منصور | دوندی | 24 |
| card_006.jpg | `birth_date` | 1365/04/27 | 1312/07/5360 | 0 |
| card_006.jpg | `serial_number` | 842172249 | 22223 | 8 |
| card_006.jpg | `gender` | مرد | — | 0 |
| card_007.jpg | `serial_number` | 289694322 | 28824 | 86 |
| card_008.jpg | `birth_date` | 1336/08/27 | 1334/08/27 | 21 |
| card_009.jpg | `birth_date` | 1359/01/01 | 1359/06/01 | 95 |
| card_009.jpg | `serial_number` | 23645293 | 244424 | 84 |
| card_010.jpg | `national_code` | 4569488803 | 03 | 86 |
| card_010.jpg | `last_name` | گودرزی | نام خانوادگی | 82 |
| card_010.jpg | `first_name` | زهرا | هرا | 86 |
| card_010.jpg | `serial_number` | 979012143 | 0135133 | 56 |
| card_011.jpg | `national_code` | 0918202655 | 96804 | 91 |
| card_011.jpg | `last_name` | رضایی | دضاد | 66 |
| card_011.jpg | `birth_date` | 1363/08/14 | 1327/08/14 | 98 |
| card_011.jpg | `gender` | زن | مرد | 21 |
| card_012.jpg | `first_name` | فرشته | اما | 41 |
| card_012.jpg | `father_name` | علی | اسنا | 36 |
| card_012.jpg | `birth_date` | 1374/11/08 | 1115/55/43 | 81 |
| card_012.jpg | `serial_number` | 4841507 | 13457 | 48 |
| card_012.jpg | `gender` | زن | ۰ | 63 |
| card_013.jpg | `first_name` | نیما | ارکم | 23 |
| card_013.jpg | `father_name` | جواد | مه | 34 |
| card_013.jpg | `birth_date` | 1375/04/19 | 1313/09/94 | 26 |
| card_013.jpg | `serial_number` | 653898989 | 65389 | 100 |
| card_014.jpg | `first_name` | آیدا | ایدا | 81 |
| card_015.jpg | `birth_date` | 1382/11/17 | 1383/12/17 | 96 |
| card_015.jpg | `serial_number` | 016786355 | 016786388 | 91 |
| card_016.jpg | `birth_date` | 1352/10/15 | 1193/19/19 | 88 |
| card_016.jpg | `serial_number` | 4435624 | 44324 | 84 |
| card_016.jpg | `gender` | زن | 3 (1۱۳ | 48 |
| card_019.jpg | `serial_number` | 82378075 | 824724 | 84 |
