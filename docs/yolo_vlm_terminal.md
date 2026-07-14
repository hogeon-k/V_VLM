# YOLO + Ollama VLM Terminal Inspection

This guide covers the single-image terminal flow, batch validation, and VLM repeatability checks for PCB defect inspection.

## Core Policy

- YOLO is the authoritative source for the final OK/NG judgment, defect class, confidence, detection order, location, and bounding box.
- The VLM is only an explanation assistant.
- The VLM receives the full YOLO result image and one detection crop montage for NG results.
- The VLM must return compact JSON matching the configured schema.
- Python parses and formats the final readable response deterministically.
- If parsing fails, the app uses a YOLO-based fallback response and preserves the raw VLM response separately.

## Deterministic Generation

The default Ollama generation options are:

```text
num_ctx = 8192
num_predict = 256
temperature = 0.0
top_p = 0.8
top_k = 20
repeat_penalty = 1.1
seed = 42
```

These options reduce output variation, but they do not guarantee exact reproducibility across different Ollama versions, model builds, quantization variants, or hardware backends.

## VLM Image Size Controls

The VLM receives two JPEG images for NG results: the full YOLO result image and one detection crop montage. Defaults are:

```text
full image longest side = 960
crop montage longest side = 960
crop tile maximum side = 512
JPEG quality = 90
```

Use these options to isolate image-size effects without changing prompt, schema, YOLO, crop padding, crop tile limits, or the two-image VLM input structure:

```text
--vlm-full-image-size 640
--vlm-montage-size 640
```

Both options preserve aspect ratio and do not upscale smaller images. Larger values can increase Ollama image token/context pressure and may cause empty or incomplete responses with schema-constrained VLM calls.

Use `--vlm-image-mode` to isolate whether the VLM failure is caused by the two-image payload:

```text
--vlm-image-mode full
--vlm-image-mode montage
--vlm-image-mode full_montage
```

The default is `full_montage`, which preserves the existing image payload behavior. In `full` mode only the full image bytes are sent to the VLM. In `montage` mode only the crop montage bytes are sent. JSON Schema, format, YOLO settings, crop parameters, generation options, and parser validation rules are unchanged by the image mode.

## Structured Output

The VLM prompt is written primarily in Korean so the model is encouraged to return Korean `visual_feature` and `summary` text. Operator-facing formatted explanations and fallback explanations are also printed in Korean.

The Ollama request still uses the `format` field with a JSON Schema. JSON keys and enum values remain in English for compatibility. The response must contain:

```text
final_judgment
detections
summary
```

Each detection must contain:

```text
detection_id
visual_feature
visibility
review_required
```

The parser rejects invalid JSON, missing fields, unknown enum values, additional properties, detection count mismatches, and detection ID mismatches. CSV column names also remain in English, while CSV values may contain Korean explanation text.

## Explanation Quality Warnings

JSON parse success only means the VLM response matched the required structure. It does not guarantee that `visual_feature` is a useful visual explanation.

After a successful parse, the parser separately records explanation quality metadata:

```text
quality_status
class_name_only_count
summary_contradiction
semantic_warning_count
```

`quality_status=warning` means the JSON is usable, but at least one explanation quality issue was detected. The current checks flag `visual_feature` values that only repeat the YOLO class name, such as `short`, `open_circuit`, or `<class> defect`, and explicit summary contradictions such as an unclear detection paired with `All defects are clearly visible.`

These warnings do not change the YOLO-authoritative judgment, parse status, fallback behavior, or detection count/ID validation.

## Ollama Response Extraction

The client uses Ollama chat responses by default and extracts the raw VLM content from:

```text
message.content
```

For compatibility with generate-style responses, it also supports:

```text
response
```

The raw content is saved before JSON parsing. If Ollama returns an error field, the client raises a clear runtime error. If the HTTP/SDK response exists but contains no assistant content, the service records:

```text
Ollama response JSON did not contain assistant content.
```

as the parse error and uses the YOLO fallback response. Use `--vlm-debug-response` to print safe response-shape diagnostics such as top-level keys, `message.content` presence, content length, `done`, and timing fields. The debug output does not print image bytes, base64 images, or the full request payload.

The current Ollama endpoint is `/api/chat`, and requests explicitly use `stream: false`. A `done=false` status is not forced to success. It is treated as retryable only when the fully-read non-streaming HTTP response has no usable `message.content` or `response` text. In that case, the failure diagnostics preserve endpoint, stream mode, HTTP status, `done`, `done_reason`, Ollama error text, content length, token counts, and duration fields.

VLM attempts are logged like this:

```text
[INFO] VLM attempt 1/3 started
[INFO] Ollama endpoint: /api/chat
[INFO] Ollama stream: false
[INFO] Ollama HTTP status: 200
[INFO] Ollama done: false
[INFO] Ollama done_reason:
[INFO] Ollama response length: 0
[INFO] VLM attempt 1/3 failed: done_false
[INFO] Retrying after 2 seconds
```

With `--save-raw-response-on-failure`, failure files under `raw_responses/` contain a JSON diagnostic envelope rather than only the final text. The envelope includes the response body when available, plus `error_type` and `error_message`.

## Single-Image Test

```powershell
.\.venv\Scripts\python.exe scripts\test_yolo_vlm.py `
  --image data\input_images\test.jpg `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.5 `
  --device 0 `
  --vlm-model qwen2.5vl:3b `
  --ollama-host http://127.0.0.1:11434 `
  --vlm-num-ctx 8192 `
  --vlm-num-predict 256 `
  --vlm-temperature 0.0 `
  --vlm-top-p 0.8 `
  --vlm-top-k 20 `
  --vlm-repeat-penalty 1.1 `
  --vlm-seed 42 `
  --vlm-full-image-size 960 `
  --vlm-montage-size 960 `
  --vlm-image-mode full_montage
```

Add `--debug-vlm` to print the raw structured response.
Add `--vlm-debug-response` to print safe Ollama response-shape diagnostics.

## Optional Crop Montage Saving

Add this flag to save the generated detection crop montage for debugging:

```powershell
--save-crop-montage
```

The default save directory is:

```text
data/result_images/montage
```

Saved file names use the source image stem plus a timestamp:

```text
test_crop_montage_20260714_183012_123456.jpg
```

Saving does not add another VLM image and does not regenerate the montage.

## Test Image Directory

Repeatable test images belong under:

```text
data/vlm_test_images/
  open_circuit/
  short/
  missing_hole/
  normal/
  low_confidence/
  multiple_defects/
  false_positive_candidates/
```

Ground truth is treated as definitive only for `open_circuit`, `short`, `missing_hole`, and `normal`.

## Batch Test

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm_test_batch.py `
  --input-dir data\vlm_test_images `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.5 `
  --device 0 `
  --vlm-model qwen2.5vl:3b `
  --ollama-host http://127.0.0.1:11434 `
  --vlm-num-ctx 8192 `
  --vlm-num-predict 256 `
  --vlm-temperature 0.0 `
  --vlm-top-p 0.8 `
  --vlm-top-k 20 `
  --vlm-repeat-penalty 1.1 `
  --vlm-seed 42 `
  --vlm-full-image-size 960 `
  --vlm-montage-size 960 `
  --vlm-image-mode full_montage `
  --save-crop-montage
```

The batch script recursively scans `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files. Uppercase extensions are supported.

Batch processing is sequential per image. The script completes YOLO, optional VLM, response validation, fallback if needed, and per-image result saving before moving to the next image. It does not run YOLO over the full folder first and then process all NG images with VLM later.

OK images, meaning YOLO produced zero detections, skip VLM and are saved with `image_status=completed` and `vlm_status=not_run`. NG images create a crop montage, use the configured `--vlm-image-mode`, call VLM, validate the response, and retry bounded failures before using fallback.

Additional batch stability options:

```powershell
--vlm-max-retries 2 `
--vlm-retry-delay 0.5 `
--vlm-timeout 120 `
--continue-on-error `
--save-raw-response-on-failure
```

`--vlm-max-retries` is the number of retries after the first VLM attempt. `--vlm-retry-delay` waits between retries. `--vlm-timeout` is passed to the Ollama HTTP client. The batch continues after image-level failures by default, while initialization errors such as a missing model can still stop the run.

Expected sequential log shape:

```text
[INFO] [1/5] normal.jpg processing started
[INFO] [1/5] normal.jpg YOLO started
[INFO] [1/5] normal.jpg YOLO completed
[INFO] [1/5] normal.jpg OK
[INFO] [1/5] normal.jpg VLM skipped
[INFO] [1/5] normal.jpg result saved
[INFO] [1/5] normal.jpg processing completed
[INFO] [2/5] open_circuit.jpg processing started
[INFO] [2/5] open_circuit.jpg YOLO started
[INFO] [2/5] open_circuit.jpg NG, detection 1 count
[INFO] [2/5] open_circuit.jpg VLM started
[INFO] [2/5] open_circuit.jpg VLM response validation completed
[INFO] [2/5] open_circuit.jpg result saved
[INFO] [2/5] open_circuit.jpg processing completed
```

## Batch CSV

Batch result CSV files are written to:

```text
data/result_images/vlm_batch_results/
```

Each image is saved immediately after it finishes. The batch output directory contains:

```text
vlm_batch_results/
  results/
  result_images/
  montage/
  raw_responses/
  batch_summary.json
  failed_images.json
  vlm_batch_results_<timestamp>.csv
```

The CSV is written with UTF-8 BOM for Excel compatibility and is rewritten after each image so completed rows remain available during a long batch. It includes the original columns plus:

```text
vlm_raw_response
vlm_parse_success
vlm_parse_error
vlm_fallback_used
vlm_temperature
vlm_top_p
vlm_top_k
vlm_repeat_penalty
vlm_seed
vlm_image_mode
vlm_image_count
crop_count
vlm_full_image_size_limit
vlm_montage_size_limit
vlm_full_image_width
vlm_full_image_height
montage_width
montage_height
quality_status
class_name_only_count
summary_contradiction
semantic_warning_count
class_name_only_detection_ids
image_status
retry_count
failure_reason
vlm_error_type
vlm_error_message
pipeline_success
yolo_success
vlm_attempted
vlm_success
result_saved
ollama_endpoint
ollama_stream
ollama_error
```

`vlm_response` stores the final user-facing response. `vlm_raw_response` stores the exact Ollama text before parsing. Multiline values are written through Python's CSV module.

For normal images with zero detections, VLM analysis is skipped and `vlm_response` is:

```text
VLM analysis skipped because no defect was detected.
```

`batch_summary.json` stores total image count, pipeline completed/failed counts, YOLO success/failed counts, OK/NG counts, VLM attempted/success/skipped counts, first-attempt success, retry success, fallback count, result save success count, final failure count, common failure counters such as `done_false`, `empty_content`, `invalid_json`, `schema_error`, `timeout`, total processing time, average image time, average VLM time, and one compact summary per image.

## Repeatability Test

```powershell
.\.venv\Scripts\python.exe scripts\test_vlm_repeatability.py `
  --image data\input_images\test.jpg `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.5 `
  --device 0 `
  --vlm-model qwen2.5vl:3b `
  --ollama-host http://127.0.0.1:11434 `
  --vlm-num-ctx 8192 `
  --vlm-num-predict 256 `
  --vlm-temperature 0.0 `
  --vlm-top-p 0.8 `
  --vlm-top-k 20 `
  --vlm-repeat-penalty 1.1 `
  --vlm-seed 42 `
  --vlm-full-image-size 960 `
  --vlm-montage-size 960 `
  --vlm-image-mode full_montage `
  --repeat-count 5
```

The script runs YOLO once, then repeats the VLM call with the same YOLO result and identical generation settings. It prints parse success count, fallback count, exact-match counts, consistency checks, and SHA-256 hashes for raw responses and canonical parsed JSON.

Exact-match results measure whether this local stack produced byte-identical raw text or canonical parsed JSON for repeated calls. A mismatch does not automatically mean the inspection result is invalid; compare parse success, detection IDs, and final judgment consistency first.
