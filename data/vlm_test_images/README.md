# VLM Test Image Set

Use these folders to keep repeatable YOLO + VLM validation images. Place each image in the folder that matches its main test purpose, even if the same source image could exercise more than one scenario.

## Directories

- `open_circuit`: confirmed Open Circuit defect images.
- `short`: confirmed Short defect images.
- `missing_hole`: confirmed Missing Hole defect images.
- `normal`: normal images without defects.
- `low_confidence`: images where YOLO confidence is expected to be low.
- `multiple_defects`: images containing more than one defect in a single image.
- `false_positive_candidates`: images likely to expose YOLO false positives.

## Recommended Counts

- At least 5 images per defect class.
- At least 3 normal images.
- At least 3 low-confidence images.
- At least 3 multiple-defect images.
- Initial total target: 20 to 30 images.

## File Names

Recommended examples:

```text
open_circuit_001.jpg
short_001.jpg
missing_hole_001.jpg
normal_001.jpg
low_confidence_001.jpg
multiple_defects_001.jpg
```

Avoid copying the same image into multiple folders unless there is a clear test reason. Prefer one primary folder based on the scenario being validated.
