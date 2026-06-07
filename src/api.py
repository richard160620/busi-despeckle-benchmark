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
from src.metrics import hd95 as _hd95
from src.metrics import iou as _iou
from src.metrics import niqe as _niqe
from src.metrics import psnr as _psnr
from src.metrics import ssim as _ssim
from src.segment import postprocess_mask, predict

_SUPPORTED_METHODS = frozenset(["none", "median", "lee", "frost", "srad", "nlm"])

_HTML_FORM = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BUSI Despeckle &amp; Segment Demo</title>
  <style>
    body{font-family:sans-serif;max-width:800px;margin:2em auto}
    #results{display:none;margin-top:1.5em}
    .images{display:flex;gap:1em;margin-top:1em}
    .images figure{flex:1;text-align:center;margin:0}
    .images img{width:100%;border:1px solid #ccc}
    .images figcaption{margin-top:.4em;font-weight:bold}
    #metrics{margin-top:1em;font-size:1.1em}
    #error{color:red;margin-top:1em}
    button{cursor:pointer}
  </style>
</head>
<body>
  <h1>Breast Ultrasound Despeckle &amp; Segment</h1>
  <form id="form" enctype="multipart/form-data">
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
  <div id="error"></div>
  <div id="results">
    <div class="images">
      <figure>
        <img id="img-original" src="" alt="Original">
        <figcaption>Original</figcaption>
      </figure>
      <figure>
        <img id="img-denoised" src="" alt="Denoised">
        <figcaption>Denoised</figcaption>
      </figure>
      <figure>
        <img id="img-mask" src="" alt="Mask">
        <figcaption>Mask</figcaption>
      </figure>
    </div>
    <div id="metrics"></div>
  </div>
  <script>
    document.getElementById('form').addEventListener('submit', async function(e) {
      e.preventDefault();
      const errorEl = document.getElementById('error');
      const resultsEl = document.getElementById('results');
      errorEl.textContent = '';
      resultsEl.style.display = 'none';

      const formData = new FormData(this);
      let data;
      try {
        const resp = await fetch('/segment', {method: 'POST', body: formData});
        data = await resp.json();
        if (!resp.ok) {
          errorEl.textContent = data.error || 'Request failed';
          return;
        }
      } catch (err) {
        errorEl.textContent = 'Network error: ' + err.message;
        return;
      }

      document.getElementById('img-original').src = 'data:image/png;base64,' + data.original;
      document.getElementById('img-denoised').src = 'data:image/png;base64,' + data.denoised;
      document.getElementById('img-mask').src = 'data:image/png;base64,' + data.mask;
      document.getElementById('metrics').innerHTML =
        '<strong>Dice:</strong> ' + data.dice + ' &nbsp; <strong>IoU:</strong> ' + data.iou;
      resultsEl.style.display = 'block';
    });
  </script>
</body>
</html>"""


def _ndarray_to_b64_png(arr: np.ndarray) -> str:
    """Convert a 2-D or 3-D numpy array to a base64-encoded PNG string."""
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
    # Image.fromarray handles 2D (L) or 3D (RGB) automatically if shape is (H,W,3)
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _create_overlay(image_float: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create a semi-transparent red mask overlay on a grayscale float [0,1] image.

    Returns:
        uint8 RGB numpy array.
    """
    # Create RGB version of grayscale image
    rgb = np.stack([image_float] * 3, axis=-1)
    # Define red color
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Blend: (1-alpha)*image + alpha*red
    overlay = rgb.copy()
    overlay[mask] = (1 - alpha) * rgb[mask] + alpha * red

    return (overlay * 255).clip(0, 255).astype(np.uint8)


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
        """Edge: POST /segment -> validate_file -> multi-process -> respond."""
        # Edge: validate_file
        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "No image file provided"}), 400

        # Get methods (handle both single string and multiple values)
        methods = request.form.getlist("method")
        if not methods:
            methods = [request.form.get("method", "none")]
        
        # Edge: validate_methods
        for m in methods:
            if m not in _SUPPORTED_METHODS:
                return jsonify({"error": f"Unknown despeckle method: '{m}'"}), 400

        # Edge: process image
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
        orig_b64 = _ndarray_to_b64_png(image)

        results = []
        for method in methods:
            # Despeckle then re-segment
            denoised_float = _despeckle(img_float, method=method, window_size=3)
            prob_denoised = predict(_model, denoised_float)
            mask_denoised = postprocess_mask(prob_denoised)

            # Segmentation metrics
            dice_val = _dice(mask_denoised, mask_orig)
            iou_val = _iou(mask_denoised, mask_orig)
            hd95_val = _hd95(mask_denoised, mask_orig)

            # Image quality metrics (relative to original)
            psnr_val = _psnr(denoised_float, img_float)
            ssim_val = _ssim(denoised_float, img_float)
            niqe_val = _niqe(denoised_float)

            overlay_uint8 = _create_overlay(denoised_float, mask_denoised)
            denoised_uint8 = (denoised_float * 255).clip(0, 255).astype(np.uint8)

            results.append({
                "method": method,
                "dice": round(float(dice_val), 4),
                "iou": round(float(iou_val), 4),
                "hd95": round(float(hd95_val), 2),
                "psnr": round(float(psnr_val), 2),
                "ssim": round(float(ssim_val), 4),
                "niqe": round(float(niqe_val), 4),
                "original": orig_b64,
                "denoised": _ndarray_to_b64_png(denoised_uint8),
                "mask": _ndarray_to_b64_png(mask_denoised),
                "overlay": _ndarray_to_b64_png(overlay_uint8),
            })

        # Backward compatibility: return single object if only one method was requested
        # unless it was sent as a list (standard form behavior for multiple selects)
        if len(methods) == 1 and "method" in request.form and not isinstance(request.form.getlist("method"), list) :
             # This logic is tricky in Flask because getlist always returns a list.
             # We'll use the test expectation: if it's a bulk request, return list.
             pass

        # If more than 1 result, or explicitly requested as multi (we'll check if 'method' key appeared once)
        # Actually, let's just use the length of the list to decide.
        # To avoid breaking old tests, if len(methods) == 1, return the dict.
        # But wait, TestAdvancedAPI.test_multi_method_returns_list_of_results sends 2 methods.
        
        if len(results) == 1:
            return jsonify(results[0])
        return jsonify(results)

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


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
