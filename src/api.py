"""Flask application factory for the BUSI despeckle demo.

Routes:
  GET  /         — HTML upload form (ISP: C1=valid route, C2=GET method)
  POST /segment  — despeckle + segment + metrics; returns JSON + base64 PNGs

Error handlers: 400 (bad request), 404 (not found), 405 (method not allowed).

Graph Coverage edges (W7):
  [START] -> route_dispatch
    -> [GET /]      -> index         -> html_response       -> [END]
    -> [POST /seg]  -> validate_file -> [missing -> 400]
                    -> validate_meth -> [unknown -> 400]
                    -> process       -> json_response        -> [END]
    -> [unknown]    -> 404_handler                           -> [END]
    -> [wrong_verb] -> 405_handler                           -> [END]
"""
import base64
import io

import numpy as np
from flask import Flask, jsonify, request

from src.despeckle import despeckle as _despeckle
from src.metrics import dice as _dice
from src.metrics import iou as _iou
from src.segment import postprocess_mask, predict

_SUPPORTED_METHODS = frozenset(["none", "median", "lee", "frost", "srad", "nlm"])

_HTML_FORM = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BUSI Despeckle &amp; Segment Demo</title>
  <style>body{{font-family:sans-serif;max-width:600px;margin:2em auto}}</style>
</head>
<body>
  <h1>Breast Ultrasound Despeckle &amp; Segment</h1>
  <form method="post" action="/segment" enctype="multipart/form-data">
    <p><label>Image (PNG/JPG):
      <input type="file" name="image" accept="image/*" required>
    </label></p>
    <p><label>Despeckle method:
      <select name="method">
        <option value="none">none</option>
        <option value="median">median</option>
        <option value="lee">lee</option>
        <option value="frost">frost</option>
        <option value="srad">srad</option>
        <option value="nlm">nlm</option>
      </select>
    </label></p>
    <p><button type="submit">Segment</button></p>
  </form>
</body>
</html>"""


def _ndarray_to_b64_png(arr: np.ndarray) -> str:
    """Convert a 2-D numpy array to a base64-encoded PNG string."""
    from PIL import Image

    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_app(model=None):
    """Flask application factory.

    Args:
        model: injectable segmentation model or mock. If None, build_model() is called.
    """
    app = Flask(__name__)

    if model is None:
        from src.segment import build_model
        _model = build_model()
    else:
        _model = model

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        """Edge: GET / -> html_response."""
        return _HTML_FORM, 200

    @app.route("/segment", methods=["POST"])
    def segment():
        """Edge: POST /segment -> validate_file -> validate_method -> process."""
        # Edge: validate_file
        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "No image file provided"}), 400

        # Edge: validate_method
        method = request.form.get("method", "none")
        if method not in _SUPPORTED_METHODS:
            return jsonify({"error": f"Unknown despeckle method: '{method}'"}), 400

        # Edge: process
        file = request.files["image"]
        try:
            from PIL import Image
            pil_img = Image.open(file.stream).convert("L")
            image = np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            return jsonify({"error": f"Cannot read image: {exc}"}), 400

        img_float = image.astype(np.float32) / 255.0

        # Baseline: segment original → use as pseudo ground-truth
        prob_orig = predict(_model, img_float)
        mask_orig = postprocess_mask(prob_orig)

        # Despeckle then re-segment
        denoised_float = _despeckle(img_float, method=method, window_size=3)
        prob_denoised = predict(_model, denoised_float)
        mask_denoised = postprocess_mask(prob_denoised)

        dice_val = _dice(mask_denoised, mask_orig)
        iou_val = _iou(mask_denoised, mask_orig)

        denoised_uint8 = (denoised_float * 255).clip(0, 255).astype(np.uint8)

        return jsonify({
            "dice": round(float(dice_val), 4),
            "iou": round(float(iou_val), 4),
            "original": _ndarray_to_b64_png(image),
            "denoised": _ndarray_to_b64_png(denoised_uint8),
            "mask": _ndarray_to_b64_png(mask_denoised),
        })

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(_e):
        """Edge: unknown_route -> 404."""
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        """Edge: wrong_verb -> 405."""
        return jsonify({"error": "Method not allowed"}), 405

    return app
