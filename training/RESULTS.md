# Results

## YOLOv8n

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 270 | 627 | 0.749 | 0.693 | 0.756 | 0.423 |
| crazing | 45 | 106 | 0.712 | 0.233 | 0.458 | 0.206 |
| inclusion | 60 | 153 | 0.749 | 0.778 | 0.807 | 0.437 |
| patches | 50 | 131 | 0.796 | 0.924 | 0.917 | 0.570 |
| pitted_surface | 45 | 63 | 0.833 | 0.746 | 0.786 | 0.516 |
| rolled-in_scale | 45 | 91 | 0.599 | 0.560 | 0.638 | 0.272 |
| scratches | 45 | 83 | 0.805 | 0.916 | 0.930 | 0.540 |

| class | AP@0.5 | AP@0.5:0.95 | precision | recall |
| :--- | ---: | ---: | ---: | ---: |
| crazing | 0.4575 | 0.2060 | 0.7120 | 0.2333 |
| inclusion | 0.8066 | 0.4366 | 0.7486 | 0.7778 |
| patches | 0.9172 | 0.5695 | 0.7961 | 0.9237 |
| pitted_surface | 0.7864 | 0.5161 | 0.8334 | 0.7460 |
| rolled-in_scale | 0.6383 | 0.2720 | 0.5991 | 0.5604 |
| scratches | 0.9301 | 0.5404 | 0.8054 | 0.9157 |

- **mAP@0.5**: 0.7560
- **mAP@0.5:0.95** : 0.4234
- **Worst-performing class** : crazing (mAP50-95: 0.2060)
- **Best-performing class** : scratches (mAP50-95: 0.5404)
- Recall is low for crazing class at 0.233

### Visible mistakes (YOLOv8n) when comparing val0_batch0_labels.jpg and val0_batch0_pred.jpg

```text
Model completely missed crazing in 4 images out of 20. Got 1 image completely wrong, the annotated image bounding box and predicted boundary box are are not overlapping. In other images, the model predicted with less number of bounding boxes than the annotated image with differences in overlap. The model did get some bounding boxes correct in some images, but they lack in the number of bounding boxes and overlap with the annotated image.
```

## YOLO26n

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 270 | 627 | 0.7 | 0.703 | 0.733 | 0.418 |
| crazing | 45 | 106 | 0.57 | 0.358 | 0.462 | 0.196 |
| inclusion | 60 | 153 | 0.731 | 0.758 | 0.81 | 0.442 |
| patches | 50 | 131 | 0.798 | 0.893 | 0.898 | 0.572 |
| pitted_surface | 45 | 63 | 0.779 | 0.726 | 0.738 | 0.508 |
| rolled-in_scale | 45 | 91 | 0.551 | 0.615 | 0.588 | 0.257 |
| scratches | 45 | 83 | 0.77 | 0.867 | 0.902 | 0.533 |

| class | AP@0.5 | AP@0.5:0.95 | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| crazing | 0.4624 | 0.1963 | 0.5697 | 0.3585 |
| inclusion | 0.8095 | 0.4421 | 0.7312 | 0.7582 |
| patches | 0.8977 | 0.5720 | 0.7980 | 0.8931 |
| pitted_surface | 0.7376 | 0.5076 | 0.7786 | 0.7257 |
| rolled-in_scale | 0.5878 | 0.2572 | 0.5512 | 0.6154 |
| scratches | 0.9021 | 0.5329 | 0.7700 | 0.8675 |

- mAP@0.5: 0.7329
- mAP@0.5:0.95: 0.4180
- Worst-performing class: crazing (mAP50-95: 0.1963)
- Best-performing class: patches (mAP50-95: 0.5720)
- Recall is low for crazing class at 0.358

### Visible mistakes (YOLO26n) when comparing val0_batch0_labels.jpg and val0_batch0_pred.jpg

```text
Model completely missed crazing in 1 image out of 20. Got 3 image wrong, the annotated image bounding box and predicted boundary box are are not overlapping a lot in terms of the bounding boxes created for both. In other images, the model predicted with less number of bounding boxes than the annotated image with differences in overlap. The model did get some bounding boxes correct in some images, but they lack in the number of bounding boxes and overlap with the annotated image.
```
