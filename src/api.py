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
import math

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
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BUSI Medical Imaging Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .img-container { position: relative; width: 100%; aspect-ratio: 1/1; background: #1a1a1a; overflow: hidden; border-radius: 0.5rem; }
    .img-container img { width: 100%; height: 100%; object-fit: contain; }
    .overlay-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
    .show-overlay .overlay-img { opacity: 1; }
    .loading-spinner { border: 4px solid rgba(255, 255, 255, 0.1); border-left-color: #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen font-sans">
  <div class="flex flex-col md:flex-row h-screen overflow-hidden">
    <!-- Sidebar -->
    <aside class="w-full md:w-80 bg-gray-800 border-r border-gray-700 p-6 flex-shrink-0 overflow-y-auto">
      <div class="flex items-center gap-3 mb-8">
        <div class="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center font-bold">B</div>
        <h1 class="text-xl font-bold tracking-tight">BUSI Dashboard</h1>
      </div>

      <form id="form" class="space-y-6">
        <div class="space-y-2">
          <label class="block text-sm font-medium text-gray-400 uppercase tracking-wider">Source Image</label>
          <div class="relative group">
            <input type="file" name="image" id="image-input" accept="image/*" required
              class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10">
            <div class="border-2 border-dashed border-gray-600 group-hover:border-blue-500 rounded-xl p-4 text-center transition-colors">
              <span id="file-name" class="text-sm text-gray-500 italic">Drop or click to upload</span>
            </div>
          </div>
        </div>

        <div class="space-y-3">
          <label class="block text-sm font-medium text-gray-400 uppercase tracking-wider">Despeckle Methods</label>
          <div class="grid grid-cols-2 gap-2">
            <label class="flex items-center gap-2 p-2 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors">
              <input type="checkbox" name="method" value="none" checked class="rounded border-gray-500 text-blue-500">
              <span class="text-sm">None</span>
            </label>
            <label class="flex items-center gap-2 p-2 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors">
              <input type="checkbox" name="method" value="median" class="rounded border-gray-500 text-blue-500">
              <span class="text-sm">Median</span>
            </label>
            <label class="flex items-center gap-2 p-2 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors">
              <input type="checkbox" name="method" value="lee" class="rounded border-gray-500 text-blue-500">
              <span class="text-sm">Lee</span>
            </label>
            <label class="flex items-center gap-2 p-2 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors">
              <input type="checkbox" name="method" value="frost" class="rounded border-gray-500 text-blue-500">
              <span class="text-sm">Frost</span>
            </label>
            <label class="flex items-center gap-2 p-2 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors">
              <input type="checkbox" name="method" value="srad" class="rounded border-gray-500 text-blue-500">
              <span class="text-sm">SRAD</span>
            </label>
            <label class="flex items-center gap-2 p-2 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors">
              <input type="checkbox" name="method" value="nlm" class="rounded border-gray-500 text-blue-500">
              <span class="text-sm">NLM</span>
            </label>
          </div>
        </div>

        <button type="submit" 
          class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-blue-900/20 transition-all flex items-center justify-center gap-2">
          <span>Run Analysis</span>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path></svg>
        </button>
      </form>

      <div class="mt-8 pt-8 border-t border-gray-700">
        <label class="flex items-center justify-between text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">
          Global Settings
        </label>
        <div class="space-y-4">
          <label class="flex items-center justify-between group cursor-pointer">
            <span class="text-sm group-hover:text-white transition-colors">Show Red Overlays</span>
            <input type="checkbox" id="toggle-overlay" class="sr-only peer">
            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600 relative"></div>
          </label>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-y-auto p-8 lg:p-12 relative">
      <div id="loading" class="hidden absolute inset-0 bg-gray-900/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center">
        <div class="loading-spinner mb-4"></div>
        <p class="text-blue-400 font-medium animate-pulse">Processing ultra-sound data...</p>
      </div>

      <div id="error" class="hidden bg-red-900/20 border border-red-500/50 text-red-200 p-4 rounded-xl mb-8 flex items-center gap-3">
        <svg class="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>
        <span id="error-text"></span>
      </div>

      <div id="welcome" class="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
        <div class="w-20 h-20 bg-gray-800 rounded-3xl flex items-center justify-center mb-8 shadow-2xl">
          <svg class="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
        </div>
        <h2 class="text-3xl font-bold mb-4">Ready to start?</h2>
        <p class="text-gray-500 leading-relaxed">Upload a breast ultrasound scan on the left sidebar and select the despeckling methods you wish to compare. The system will automatically detect lesions and calculate performance metrics.</p>
      </div>

      <div id="results" class="hidden space-y-12 pb-12">
        <header class="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h2 class="text-2xl font-bold">Analysis Results</h2>
            <p class="text-gray-500">Comparison across selected despeckling filters</p>
          </div>
          <div class="flex gap-4">
            <div class="px-4 py-2 bg-gray-800 rounded-lg border border-gray-700">
              <span class="text-xs text-gray-500 block uppercase">Input</span>
              <span id="results-count" class="font-semibold">0 methods</span>
            </div>
          </div>
        </header>

        <div id="results-grid" class="grid grid-cols-1 xl:grid-cols-2 gap-8">
          <!-- Dynamic Results Here -->
        </div>

        <section class="mt-12 bg-gray-800 border border-gray-700 rounded-2xl overflow-hidden shadow-2xl">
          <div class="px-6 py-4 bg-gray-750 border-b border-gray-700">
            <h3 class="font-bold">Metrics Summary Table</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-sm">
              <thead class="bg-gray-750 text-gray-400 font-medium">
                <tr>
                  <th class="px-6 py-3 border-b border-gray-700">Method</th>
                  <th class="px-6 py-3 border-b border-gray-700">Dice ↑</th>
                  <th class="px-6 py-3 border-b border-gray-700">IoU ↑</th>
                  <th class="px-6 py-3 border-b border-gray-700">HD95 ↓</th>
                  <th class="px-6 py-3 border-b border-gray-700">PSNR ↑</th>
                  <th class="px-6 py-3 border-b border-gray-700">SSIM ↑</th>
                  <th class="px-6 py-3 border-b border-gray-700">NIQE ↓</th>
                </tr>
              </thead>
              <tbody id="metrics-table-body" class="divide-y divide-gray-700">
                <!-- Dynamic Rows Here -->
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  </div>

  <template id="result-template">
    <div class="bg-gray-800 border border-gray-700 rounded-2xl p-6 space-y-6 shadow-xl hover:border-gray-600 transition-colors">
      <div class="flex items-center justify-between border-b border-gray-700 pb-4">
        <h3 class="text-lg font-bold capitalize method-name">Method</h3>
        <div class="flex gap-3 text-xs font-mono">
          <span class="bg-blue-500/20 text-blue-400 px-2 py-1 rounded">Dice: <span class="val-dice">0</span></span>
          <span class="bg-green-500/20 text-green-400 px-2 py-1 rounded">IoU: <span class="val-iou">0</span></span>
        </div>
      </div>
      
      <div class="grid grid-cols-2 gap-4">
        <div class="space-y-2">
          <span class="text-xs text-gray-500 uppercase">Denoised Result</span>
          <div class="img-container group relative">
            <img class="img-denoised" src="" alt="Denoised">
            <img class="overlay-img" src="" alt="Overlay">
            <div class="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
               <span class="bg-black/60 backdrop-blur px-2 py-1 rounded text-[10px]">DENOISED</span>
            </div>
          </div>
        </div>
        <div class="space-y-2">
          <span class="text-xs text-gray-500 uppercase">Detected Mask</span>
          <div class="img-container bg-black">
            <img class="img-mask" src="" alt="Mask">
          </div>
        </div>
      </div>

      <div class="grid grid-cols-3 gap-2">
         <div class="bg-gray-900/50 p-2 rounded-lg text-center">
            <span class="block text-[10px] text-gray-500 uppercase">PSNR</span>
            <span class="text-sm font-bold val-psnr">0</span>
         </div>
         <div class="bg-gray-900/50 p-2 rounded-lg text-center">
            <span class="block text-[10px] text-gray-500 uppercase">SSIM</span>
            <span class="text-sm font-bold val-ssim">0</span>
         </div>
         <div class="bg-gray-900/50 p-2 rounded-lg text-center">
            <span class="block text-[10px] text-gray-500 uppercase">NIQE</span>
            <span class="text-sm font-bold val-niqe">0</span>
         </div>
      </div>
    </div>
  </template>

  <script>
    const form = document.getElementById('form');
    const imageInput = document.getElementById('image-input');
    const fileName = document.getElementById('file-name');
    const toggleOverlay = document.getElementById('toggle-overlay');
    const resultsEl = document.getElementById('results');
    const welcomeEl = document.getElementById('welcome');
    const loadingEl = document.getElementById('loading');
    const errorEl = document.getElementById('error');
    const errorText = document.getElementById('error-text');
    const resultsGrid = document.getElementById('results-grid');
    const metricsTableBody = document.getElementById('metrics-table-body');
    const resultsCount = document.getElementById('results-count');
    const resultTemplate = document.getElementById('result-template');

    imageInput.addEventListener('change', (e) => {
      fileName.textContent = e.target.files[0] ? e.target.files[0].name : 'Drop or click to upload';
    });

    toggleOverlay.addEventListener('change', (e) => {
      document.body.classList.toggle('show-overlay', e.target.checked);
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      errorEl.classList.add('hidden');
      loadingEl.classList.remove('hidden');
      
      const formData = new FormData(form);
      
      try {
        const resp = await fetch('/segment', { method: 'POST', body: formData });
        let data = await resp.json();
        
        if (!resp.ok) throw new Error(data.error || 'Analysis failed');
        
        // Ensure data is always an array for consistent rendering
        const results = Array.isArray(data) ? data : [data];
        renderResults(results);
        
        welcomeEl.classList.add('hidden');
        resultsEl.classList.remove('hidden');
      } catch (err) {
        errorText.textContent = err.message;
        errorEl.classList.remove('hidden');
      } finally {
        loadingEl.classList.add('hidden');
      }
    });

    function renderResults(results) {
      resultsGrid.innerHTML = '';
      metricsTableBody.innerHTML = '';
      resultsCount.textContent = results.length + (results.length === 1 ? ' method' : ' methods');

      results.forEach(res => {
        // Grid item
        const card = resultTemplate.content.cloneNode(true);
        card.querySelector('.method-name').textContent = res.method;
        card.querySelector('.val-dice').textContent = res.dice;
        card.querySelector('.val-iou').textContent = res.iou;
        card.querySelector('.val-psnr').textContent = res.psnr;
        card.querySelector('.val-ssim').textContent = res.ssim;
        card.querySelector('.val-niqe').textContent = res.niqe;
        
        card.querySelector('.img-denoised').src = 'data:image/png;base64,' + res.denoised;
        card.querySelector('.img-mask').src = 'data:image/png;base64,' + res.mask;
        card.querySelector('.overlay-img').src = 'data:image/png;base64,' + res.overlay;
        
        resultsGrid.appendChild(card);

        // Table row
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-700/50 transition-colors';
        row.innerHTML = `
          <td class="px-6 py-4 font-bold capitalize">${res.method}</td>
          <td class="px-6 py-4">${res.dice}</td>
          <td class="px-6 py-4">${res.iou}</td>
          <td class="px-6 py-4 text-gray-400">${res.hd95}</td>
          <td class="px-6 py-4 text-gray-400">${res.psnr}</td>
          <td class="px-6 py-4 text-gray-400">${res.ssim}</td>
          <td class="px-6 py-4 text-gray-400">${res.niqe}</td>
        `;
        metricsTableBody.appendChild(row);
      });
    }
  </script>
</body>
</html>"""


def _finite_round(value: float, ndigits: int, cap: float = 100.0) -> float:
    """Round a metric value, replacing non-finite results with a finite cap.

    PSNR is mathematically infinite when two images are identical (MSE == 0,
    e.g. the "none" despeckle method on the mock pipeline). `Infinity`/`NaN`
    are valid Python floats but are *not* valid JSON tokens — Flask's
    jsonify would emit them verbatim and the browser's resp.json() then
    throws a SyntaxError ("The string did not match the expected pattern").
    Capping keeps the response strictly valid JSON while preserving the
    "very high quality" signal for display.
    """
    value = float(value)
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        value = cap if value > 0 else -cap
    return round(value, ndigits)


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
                "psnr": _finite_round(psnr_val, 2),
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
