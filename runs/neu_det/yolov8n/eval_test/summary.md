# Evaluation summary

| field | value |
| --- | --- |
| weights | `D:\Projects\Python Projects\EdgeLens\runs\neu_det\yolov8n\weights\best.pt` |
| data | `D:\Projects\Python Projects\EdgeLens\data\neu_det\data.yaml` |
| split | test |
| images | 270 |
| imgsz | 640 |
| ultralytics | 8.4.126 |
| evaluated (UTC) | 2026-08-23T13:26:58+00:00 |

## Detection metrics (Ultralytics validator)

| metric | value |
| --- | --- |
| **mAP@0.5** | **0.7560** |
| **mAP@0.5:0.95** | **0.4234** |
| mean precision | 0.7491 |
| mean recall | 0.6928 |

## Per-class AP

| class | AP@0.5 | AP@0.5:0.95 | precision | recall |
| --- | --- | --- | --- | --- |
| crazing | 0.4575 | 0.2060 | 0.7120 | 0.2333 |
| inclusion | 0.8066 | 0.4366 | 0.7486 | 0.7778 |
| patches | 0.9172 | 0.5695 | 0.7961 | 0.9237 |
| pitted_surface | 0.7864 | 0.5161 | 0.8334 | 0.7460 |
| rolled-in_scale | 0.6383 | 0.2720 | 0.5991 | 0.5604 |
| scratches | 0.9301 | 0.5404 | 0.8054 | 0.9157 |

## Error analysis

Single operating point: conf >= 0.250, a prediction matches ground truth at IoU >= 0.50 with the same class.

| quantity | value |
| --- | --- |
| ground-truth boxes | 627 |
| predictions | 696 |
| true positives | 462 |
| false positives | 234 |
| false negatives | 165 |
| precision @ this point | 0.6638 |
| recall @ this point | 0.7368 |

### False positives by cause

| cause | count |
| --- | --- |
| background | 92 |
| duplicate | 40 |
| poor_localisation | 101 |
| wrong_class | 1 |

### False negatives by cause

| cause | count |
| --- | --- |
| missed_entirely | 80 |
| poor_localisation | 84 |
| wrong_class | 1 |

### Per class

| class | false positives | false negatives |
| --- | --- | --- |
| crazing | 37 | 73 |
| inclusion | 59 | 27 |
| patches | 39 | 10 |
| pitted_surface | 13 | 16 |
| rolled-in_scale | 57 | 33 |
| scratches | 29 | 6 |

Annotated worst cases are in `false_positives/` and `false_negatives/`; the full
lists (not just the top 20) are in `error_analysis.json`.
