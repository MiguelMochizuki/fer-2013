# Browser demo

`web/` is a static site that runs the same pipeline in the browser: YuNet finds the faces, the ONNX classifier predicts the emotion with the calibrated probabilities, and Grad-CAM++ is computed in JavaScript. The photo never leaves the page. After the models load there are no network requests (the Network tab shows it). A Content-Security-Policy restricts the page itself; the worker that processes the pixels is same-origin code with no network calls, but GitHub Pages cannot send CSP headers for workers, so the browser does not enforce that part.

How it works:
- **Live camera** ("Use your camera"): the video is mirrored on the page and analyzed as fast as the worker allows, dropping frames instead of queuing them. Every detected face is classified on every analysis; faces keep their number between frames (matched by box overlap, `web/src/live.js`) and their probabilities and boxes are smoothed with an exponential average, because the classifier is sensitive to a few pixels of crop and would otherwise flicker. A face that is missed for a few frames is kept, dimmed, instead of coming back with a new number. Measured in Chromium on a desktop with one WASM thread: about 7 analyses per second with one face, 4 with three and 3 with five. The camera frames stay in the tab like photos do, and the stream stops when you press the button, pick a photo or leave. The model was trained on acted, 48x48 grayscale faces, so a webcam in poor light or at an angle will be read worse than the test set suggests (and `neutral` tends to win).
- With several faces, a strip of crops lets you pick one at a time (or click its box on the photo); the reading, crop and heatmap follow the selection.
- Plain ES modules, no bundler. `onnxruntime-web` runs in a Web Worker with WebAssembly and one thread, because GitHub Pages cannot send the headers that threads need.
- Every stage is tested in Node against golden files written by the Python pipeline (`scripts/make_web_golden.py`): resampling is byte-identical to Pillow, detector boxes match OpenCV (IoU at least 0.999), probabilities agree within 1e-2 and Grad-CAM++ maps within 2e-2 (the int8 classifier rounds a few activations differently in WebAssembly, see [Quantization](results.md#quantization)). A Python test fails when the golden files go stale, and the site build fails when the models differ from the ones the golden files were made for.
- The models are downloaded on first use, checked against their sha256 and kept in the browser cache, so later visits do not download them again.
- Measured in Chromium on a desktop, with one face: detection about 11 ms, classification about 125 ms, Grad-CAM++ about 4 ms; roughly 100 ms more per extra face. A first visit downloads about 26 MB (the 11 MB model and the 14 MB runtime).
- PNGs with transparency: the browser's canvas stores premultiplied alpha, so fully transparent pixels reach the model as black, whereas Pillow keeps the stored RGB. Opaque images are unaffected.
- A JPEG decoded by Chromium gives probabilities that differ negligibly from Pillow's on the example photo (the golden tests allow 1e-2), and the same box. Other browsers may decode JPEG slightly differently.

Build and run it locally:

```bash
cd web && npm ci && npm test && cd ..
RELEASE_URL=<models release url> scripts/fetch_models.sh models serving/models.sha256
node web/build.mjs --models models --out web/dist
python3 -m http.server -d web/dist 8000
```

The site is published to GitHub Pages from `main` by `.github/workflows/pages.yml`. Pages is free for public repositories, with a soft limit of 100 GB of bandwidth per month, which is on the order of 1,500 first visits.

## Limitations and privacy

- The classifier reaches about 86% accuracy on the FER+ test labels and 61% on the original FER-2013 labels (see [results](results.md)), its confidences are calibrated on FER+'s validation split only, and it inherits the dataset's biases: acted or web-scraped expressions, uneven demographics, noisy labels. `disgust` and `fear` are rare in training and under-predicted. Emotion labels from a face are not a reliable read of how someone feels. Do not use this for decisions about people.
- The confidence moves with the crop, usually little: on the 1,480 upscaled FER+ test faces above, shifting the detector box by 2 px (faces about 128 px wide) changes the top confidence by 0.2 points on average and by up to 29 points in the worst case, and changes the predicted emotion in 0.3% of 5,920 shifts. Photos with a hand, glasses or a tilted head will move more than these frontal faces.
- Faces from a detector are cropped square with a 10% margin before classification. On the first 1,500 FER+ test faces upscaled 4x (1,480 detected, 98.7%), classifying the detector crop scores 85.2% against 85.3% for the original 48x48 image on the same faces, and margins from 0% to 40% all land between 85.0% and 85.2%. So the crop costs about 0.1 points and the margin hardly matters. This is a proxy built from FER faces (the first 1,500 test images, not a random sample), not a benchmark on real photos.
- Uploaded images are processed in memory and never written to disk or logged. Uploads are limited to 5 MB and JPEG, PNG or WebP.
- The service is public and unauthenticated; it caps concurrent work and answers `503` when busy.
- Neither the API nor the browser demo caps the number of faces: every face the detector finds is classified, so a crowd takes proportionally longer (the browser classifies 16 faces per batch). Detection runs on the image downscaled to at most 640 px on the long side, so small faces in a large crowd photo can be missed.
