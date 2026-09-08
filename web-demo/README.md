# web-demo — client-side NEU-DET defect detection

A single static page that runs the quantised YOLO26-nano detector entirely in the browser
with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). No backend, no upload,
no network traffic after the model downloads.

```
web-demo/
├── index.html          # page + styles
├── app.js              # preprocessing, inference, box rendering
├── model/
│   └── yolo26n_int8.onnx   # 2.80 MB
├── examples/           # 6 one-click images
├── reference.json      # Python detections, for the parity check
└── verify_parity.py    # generates reference.json; --check compares a browser export
```

Run it locally with any static server — `file://` will not work, because `fetch()` of the
model and the WASM binaries needs a real origin:

```bash
python -m http.server 8123 --directory web-demo
```

## No NMS, and why that is not an oversight

The exported graph's output is **`[1, 300, 6]`**:

```
$ python -c "import onnx; m=onnx.load('web-demo/model/yolo26n_int8.onnx'); print(m.graph.output)"
output0  [1, 300, 6]
```

300 detections of `[x1, y1, x2, y2, confidence, class_id]`, already filtered and sorted.
YOLO26's head is end-to-end: a learned one-to-one assignment replaces non-maximum
suppression, so the graph emits final boxes. `app.js` therefore contains **no NMS**, no
anchor decoding, no transpose and no sigmoid — it thresholds on confidence and maps
coordinates back through the letterbox.

This was checked before any JavaScript was written, because the alternative export is not
interchangeable: `yolov8n_int8.onnx` outputs `[1, 10, 8400]` — 4 box coordinates plus 6
class scores across 8400 anchors, raw. Shipping that model instead would make NMS in
JavaScript **mandatory**, along with anchor decoding and a transpose.

`yolo26n_int8.onnx` is also the smaller of the two INT8 exports (2.80 MB vs 3.23 MB).

## Preprocessing

Mirrors `ultralytics.data.augment.LetterBox` with the defaults a fixed-size ONNX export
uses (`auto=False`, `scale_fill=False`, `scaleup=True`, `center=True`,
`padding_value=114`, `INTER_LINEAR`):

```
r          = min(640/h, 640/w)
new_unpad  = (round(w*r), round(h*r))
dw, dh     = (640 - new_w)/2, (640 - new_h)/2
left, top  = round(dw - 0.1), round(dh - 0.1)
right, bot = round(dw + 0.1), round(dh + 0.1)
pad value  = 114
tensor     = RGB, CHW, /255, float32, shape (1,3,640,640)
```

Two details that are easy to get wrong and produce no error when you do:

- **No BGR swap.** The Python path reads BGR through OpenCV and swaps to RGB. A canvas
  gives RGBA directly, so the browser must *not* swap — both end up RGB.
- **The resize is done by hand, not by `drawImage`.** Canvas scaling quality is
  implementation-defined; browsers may use bicubic or Lanczos. That would put a
  browser-dependent difference into the tensor before inference. `resizeBilinear()` does
  the interpolation explicitly with OpenCV's pixel-centre convention,
  `src = (dst + 0.5) * scale - 0.5`.

## Parity against the Python pipeline

`verify_parity.py` first proves the Python side of the comparison is itself correct: it
reimplements the preprocessing above, runs the ONNX graph through raw onnxruntime, and
asserts the detections match what Ultralytics' own pipeline produces.

```
patches_155.jpg             7 det  max coord delta vs ultralytics 0.000503 px
scratches_141.jpg           8 det  max coord delta vs ultralytics 0.000498 px
cast_def_0_127.jpeg         1 det  max coord delta vs ultralytics 0.000500 px
...
worst coordinate difference across all images: 0.000503 px
```

Then the browser is compared against that reference. Run `window.__runParity()` in the
console, save the JSON, and:

```bash
python web-demo/verify_parity.py --check browser_output.json
```

Measured on Chrome 148 / onnxruntime-web 1.20.1 (wasm):

| image | Python | browser | max Δ per coordinate | max Δ confidence |
|---|---:|---:|---:|---:|
| `cast_def_0_127.jpeg` | 1 | 1 | 2.93 px | 0.000000 |
| `cast_def_0_240.jpeg` | 2 | 2 | 1.46 px | 0.000000 |
| `cast_ok_0_119.jpeg` | 1 | 1 | 4.39 px | 0.189 |
| `patches_155.jpg` | 7 | 8 | 1.14 px | 0.149 |
| `pitted_surface_237.jpg` | 4 | 4 | 2.86 px | 0.064 |
| `scratches_141.jpg` | 8 | 5 | 1.77 px | 0.133 |

**Worst matched-box difference: 4.39 px** on a 512 px image (0.9% of the width). Two
images also differ in *how many* detections cross the threshold.

Detections are paired by IoU, not by array index. INT8 quantises the score output onto a
small discrete set — `0.910737`, `0.704954`, `0.370146` recur across unrelated images — so
several detections carry *identical* confidences and their sort order is arbitrary.
Comparing `detection[i]` to `detection[i]` under those ties compares unrelated objects; it
reported 460 px differences before the matching was fixed.

### Where the residual comes from

Not from the JavaScript. Three measurements, in order:

1. **The JPEG decoders agree exactly.** Browser canvas and `cv2.imread` produce an
   identical channel sum (18,208,230 over 120,000 subpixels) for `patches_155.jpg`.
2. **The resize differs by ±1.** `cv2.INTER_LINEAR` on 8-bit input uses an internal
   fixed-point path; the float bilinear in `app.js` matches OpenCV's *float32* path
   instead. Result: **146,934 of 1,228,800 subpixels differ, every one of them by exactly
   1**, mean absolute difference 0.12.
3. **INT8 amplifies that ±1; FP32 does not.** Feeding the same model two tensors differing
   only by that ±1 — Python versus Python, no browser involved:

   | model | detection counts | max matched box drift |
   |---|---|---:|
   | `yolo26n_int8` | change (7→6, 8→7) | **5.49 px** |
   | `yolo26n_fp32` | identical | **0.86 px** |

So the browser sits inside the INT8 model's own sensitivity band. The `--tolerance`
default of 6 px is taken from that measured band rather than chosen to make the check
pass; on the FP32 export the same code would agree to under a pixel.

This is worth knowing beyond the demo: it is a second cost of this INT8 export, alongside
the finding in [`../benchmarks/results_aggregate.md`](../benchmarks/results_aggregate.md)
that it runs ~2.3× *slower* than FP32 on CPU. Swapping `MODEL_URL` in `app.js` to an FP32
export (9.35 MB) would give near-exact parity at 3.3× the download.

Closing the last ±1 would mean reimplementing OpenCV's fixed-point resize in JavaScript.
That is possible — a 1-D prototype matches bit-exactly — but the 2-D two-pass accumulator
did not, and the behaviour is an undocumented internal that varies across OpenCV versions
(this was measured against 5.0.0). It was not pursued.

## Examples

Three are NEU-DET test-split images the detector was built for. **Three are casting parts
from the repo's earlier binary classifier and are marked "out of domain"** — this model has
never seen a casting, so the boxes it draws on them are meaningless. They are included
because they were requested, and labelled so nobody reads them as a result.
