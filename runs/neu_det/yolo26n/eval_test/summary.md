# Evaluation summary

| field | value |
| --- | --- |
| weights | `D:\Projects\Python Projects\EdgeLens\runs\neu_det\yolo26n\weights\best.pt` |
| data | `D:\Projects\Python Projects\EdgeLens\data\neu_det\data.yaml` |
| split | test |
| images | 270 |
| imgsz | 640 |
| ultralytics | 8.4.126 |
| evaluated (UTC) | 2026-08-23T13:27:48+00:00 |

## Detection metrics (Ultralytics validator)

| metric | value |
| --- | --- |
| **mAP@0.5** | **0.7329** |
| **mAP@0.5:0.95** | **0.4180** |
| mean precision | 0.6998 |
| mean recall | 0.7031 |

## Per-class AP

| class | AP@0.5 | AP@0.5:0.95 | precision | recall |
| --- | --- | --- | --- | --- |
| crazing | 0.4624 | 0.1963 | 0.5697 | 0.3585 |
| inclusion | 0.8095 | 0.4421 | 0.7312 | 0.7582 |
| patches | 0.8977 | 0.5720 | 0.7980 | 0.8931 |
| pitted_surface | 0.7376 | 0.5076 | 0.7786 | 0.7257 |
| rolled-in_scale | 0.5878 | 0.2572 | 0.5512 | 0.6154 |
| scratches | 0.9021 | 0.5329 | 0.7700 | 0.8675 |

## Error analysis

Single operating point: conf >= 0.250, a prediction matches ground truth at IoU >= 0.50 with the same class.

| quantity | value |
| --- | --- |
| ground-truth boxes | 627 |
| predictions | 615 |
| true positives | 442 |
| false positives | 173 |
| false negatives | 185 |
| precision @ this point | 0.7187 |
| recall @ this point | 0.7049 |

### False positives by cause

| cause | count |
| --- | --- |
| background | 53 |
| duplicate | 36 |
| poor_localisation | 82 |
| wrong_class | 2 |

### False negatives by cause

| cause | count |
| --- | --- |
| missed_entirely | 92 |
| poor_localisation | 92 |
| wrong_class | 1 |

### Per class

| class | false positives | false negatives |
| --- | --- | --- |
| crazing | 25 | 68 |
| inclusion | 40 | 37 |
| patches | 29 | 14 |
| pitted_surface | 13 | 18 |
| rolled-in_scale | 45 | 36 |
| scratches | 21 | 12 |

Annotated worst cases are in `false_positives/` and `false_negatives/`; the full
lists (not just the top 20) are in `error_analysis.json`.
