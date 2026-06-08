"""Tests for src/api.py — Flask endpoint tests.

Testing methods applied:
  - Input Space Partitioning (W6): route × method × file presence × despeckle method
  - Graph Coverage (W7): every route + error-handler edge
  - Boundary conditions: missing file, unknown method, wrong HTTP verb

Status codes covered: 200, 400 (missing file), 400 (unknown method), 404, 405.
All tests inject a mock model — no GPU or real model required.
"""
import io
import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _app():
    from src.api import create_app
    import torch

    class _ModuleMock:
        def __call__(self, x):
            return torch.full_like(x, 0.8)

    app = create_app(model=_ModuleMock())
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(_app):
    return _app.test_client()


@pytest.fixture(scope="module")
def tiny_png_bytes():
    """32×32 grayscale PNG as bytes for multipart upload."""
    from PIL import Image
    img = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, (32, 32), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GET / — HTML form
# Graph edge: [START] -> index -> [END]
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_get_returns_200(self, client):
        """ISP C1/C2: valid route, valid method → 200."""
        r = client.get("/")
        assert r.status_code == 200

    def test_response_is_html(self, client):
        r = client.get("/")
        assert b"<form" in r.data

    def test_response_contains_method_select(self, client):
        r = client.get("/")
        assert b"select" in r.data

    def test_response_contains_all_methods(self, client):
        r = client.get("/")
        for method in [b"none", b"median", b"lee", b"frost", b"srad", b"nlm"]:
            assert method in r.data

    def test_response_contains_tailwind_and_dashboard_elements(self, client):
        """Test that the advanced UI elements are present."""
        r = client.get("/")
        assert b"tailwindcss.com" in r.data
        assert b"BUSI Dashboard" in r.data
        assert b"Show Red Overlays" in r.data
        assert b"Metrics Summary Table" in r.data


# ---------------------------------------------------------------------------
# POST /segment — happy path
# Graph edge: validate_file -> validate_method -> process -> respond
# ---------------------------------------------------------------------------

class TestSegmentHappyPath:
    def test_valid_request_returns_200(self, client, tiny_png_bytes):
        """ISP C1: valid image + known method → 200."""
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200

    def test_response_json_has_required_keys(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert "dice" in data
        assert "iou" in data
        assert "original" in data
        assert "denoised" in data
        assert "mask" in data

    def test_dice_iou_in_unit_interval(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert 0.0 <= data["dice"] <= 1.0
        assert 0.0 <= data["iou"] <= 1.0

    def test_base64_images_are_non_empty(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert len(data["original"]) > 0
        assert len(data["denoised"]) > 0
        assert len(data["mask"]) > 0

    @pytest.mark.parametrize("method", ["none", "median", "lee", "frost", "srad", "nlm"])
    def test_all_despeckle_methods_accepted(self, client, tiny_png_bytes, method):
        """Verify each method in _SUPPORTED_METHODS is actually accepted."""
        r = client.post(
            "/segment",
            data={"method": method, "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /segment — error paths
# Graph edges: missing_file -> 400, unknown_method -> 400
# ---------------------------------------------------------------------------

class TestSegmentErrors:
    def test_missing_image_returns_400(self, client):
        """ISP C1 invalid: no file uploaded → 400."""
        r = client.post(
            "/segment",
            data={"method": "none"},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400

    def test_missing_image_error_message(self, client):
        r = client.post(
            "/segment",
            data={"method": "none"},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert "error" in data

    def test_unknown_method_returns_400(self, client, tiny_png_bytes):
        """ISP C3 invalid: unknown despeckle method → 400."""
        r = client.post(
            "/segment",
            data={
                "method": "wavelet_super_filter",
                "image": (io.BytesIO(tiny_png_bytes), "test.png"),
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400

    def test_unknown_method_error_message(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={
                "method": "bad",
                "image": (io.BytesIO(tiny_png_bytes), "test.png"),
            },
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert "error" in data

    @pytest.mark.parametrize("bad_method", ["", "MEDIAN", "Nlm", "nlm2", "123"])
    def test_invalid_method_names_return_400(self, client, tiny_png_bytes, bad_method):
        """BVA: edge cases for method name — empty, wrong case, trailing digit."""
        r = client.post(
            "/segment",
            data={
                "method": bad_method,
                "image": (io.BytesIO(tiny_png_bytes), "test.png"),
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Advanced API features (TDD Red Stage)
# ---------------------------------------------------------------------------

class TestAdvancedAPI:
    """Tests for advanced features: multi-method, rich metrics, and overlays."""

    def test_multi_method_unknown_raises_400(self, client, tiny_png_bytes):
        """Test that if one of the multiple methods is unknown, it returns 400."""
        r = client.post(
            "/segment",
            data={
                "method": ["median", "invalid_filter"],
                "image": (io.BytesIO(tiny_png_bytes), "test.png")
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        data = r.get_json()
        assert "Unknown despeckle method" in data["error"]

    def test_multi_method_returns_list_of_results(self, client, tiny_png_bytes):
        """Test that passing multiple methods returns a list of results."""
        r = client.post(
            "/segment",
            data={
                "method": ["median", "lee"],
                "image": (io.BytesIO(tiny_png_bytes), "test.png")
            },
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["method"] == "median"
        assert data[1]["method"] == "lee"

    def test_response_contains_rich_metrics(self, client, tiny_png_bytes):
        """Test that the response contains HD95, PSNR, SSIM, and NIQE."""
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        # If it returns a list now, check first item
        result = data[0] if isinstance(data, list) else data
        assert "hd95" in result
        assert "psnr" in result
        assert "ssim" in result
        assert "niqe" in result

    def test_response_contains_overlay(self, client, tiny_png_bytes):
        """Test that the response contains a base64 overlay image."""
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        result = data[0] if isinstance(data, list) else data
        assert "overlay" in result
        assert len(result["overlay"]) > 0


# ---------------------------------------------------------------------------
# Error handlers: 404, 405
# Graph edges: unknown_route -> 404, wrong_verb -> 405
# ---------------------------------------------------------------------------

class TestErrorHandlers:
    def test_unknown_route_get_returns_404(self, client):
        """Graph edge: unknown route → 404."""
        assert client.get("/nonexistent").status_code == 404

    def test_unknown_route_post_returns_404(self, client):
        assert client.post("/nonexistent").status_code == 404

    def test_segment_get_returns_405(self, client):
        """Graph edge: GET on POST-only route → 405."""
        assert client.get("/segment").status_code == 405

    def test_segment_put_returns_405(self, client):
        assert client.put("/segment").status_code == 405

    def test_segment_delete_returns_405(self, client):
        assert client.delete("/segment").status_code == 405


# ---------------------------------------------------------------------------
# Regression: /segment JSON contract
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestSegmentResponseContract:
    """Regression guard: /segment must always return the five expected JSON keys."""

    def test_segment_returns_all_expected_keys(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert set(data.keys()) >= {"original", "denoised", "mask", "dice", "iou"}

    def test_segment_image_keys_are_nonempty_strings(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        for key in ("original", "denoised", "mask"):
            assert isinstance(data[key], str) and len(data[key]) > 0

    def test_segment_metric_keys_are_floats(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert isinstance(data["dice"], float)
        assert isinstance(data["iou"], float)


# ---------------------------------------------------------------------------
# Regression: _ndarray_to_b64_png numeric normalization (BVA: mx > mn branch)
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestNdarrayToB64PngNormalization:
    """Float arrays must be linearly normalized via division (arr-mn)/(mx-mn),
    not multiplication — kills a survived mutant in the mx > mn branch."""

    def test_float_array_is_linearly_normalized_to_uint8_range(self):
        import base64
        from PIL import Image
        from src.api import _ndarray_to_b64_png

        arr = np.array([[0.0, 5.0], [10.0, 20.0]], dtype=np.float64)
        decoded = np.array(Image.open(io.BytesIO(base64.b64decode(_ndarray_to_b64_png(arr)))))

        mn, mx = arr.min(), arr.max()
        expected = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        assert np.array_equal(decoded, expected)


# ---------------------------------------------------------------------------
# Regression: /segment must always return strictly valid JSON
# (BVA: method="none" -> denoised == original -> MSE == 0 -> PSNR == inf,
#  and `Infinity`/`NaN` are not valid JSON tokens — they break the
#  browser's resp.json() with "The string did not match the expected pattern")
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestSegmentResponseIsStrictJSON:
    def test_response_body_contains_no_non_finite_json_tokens(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        raw = r.get_data(as_text=True)
        assert "Infinity" not in raw
        assert "NaN" not in raw

    def test_psnr_is_finite_when_denoised_equals_original(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        assert math.isfinite(data["psnr"])


# ===========================================================================
# Mutation killers (W9 Syntax-Based / Mutation Testing) — see mutmut survivors
# for src/api.py. Each test below targets one or more surviving mutants
# identified via `mutmut results` + `mutmut show <id>`.
#
# Documented EQUIVALENT mutants (no test can distinguish them):
#   12, 13 — `value > 0` -> `value >= 0` / `value > 1` inside the
#            `math.isinf(value)` branch of _finite_round: value is always
#            +inf or -inf there (never 0 or 1), so all three comparisons
#            select the same branch (cap vs -cap) for every reachable input.
#   76     — `request.form.get("method", ...)` -> `.get("XXmethodXX", ...)`:
#            this fallback line only runs when `request.form.getlist("method")`
#            is empty, which (for Flask's MultiDict) only happens when the
#            "method" key is entirely absent — so `.get("method", d)` and
#            `.get("XXmethodXX", d)` both fall through to the default `d`.
#   22      — `mx > mn` -> `mx >= mn` in `_ndarray_to_b64_png`. The `>=`
#            branch only differs from `>` when mx == mn (constant array).
#            For a constant float array, arr-mn = 0 everywhere, so
#            (arr-mn)/(mx-mn) = 0/0 = NaN; NaN*255 = NaN; NaN.astype(uint8)
#            casts to 0 (with RuntimeWarning, but pytest.ini has no
#            filterwarnings=error). Result is all-zeros, identical to
#            `zeros_like`. No functional test can distinguish the branches.
#   131-137 — the `if len(methods) == 1 and "method" in request.form and
#            not isinstance(...)` block has a `pass` body (dead code, per
#            the surrounding comments admitting the logic was abandoned).
#            `request.form.getlist` has no side effects, so mutating any
#            part of this condition changes nothing observable.
# ===========================================================================

class TestApiMutationKillers:
    # -- _finite_round: cap value and branch selection (9, 11, 14) --
    def test_finite_round_caps_positive_infinity_at_100(self):
        from src.api import _finite_round
        assert _finite_round(float("inf"), 2) == 100.0

    def test_finite_round_caps_negative_infinity_at_minus_100(self):
        """Kills mutant 14 (`-cap` -> `+cap`): -inf must map to -100.0,
        not +100.0."""
        from src.api import _finite_round
        assert _finite_round(float("-inf"), 2) == -100.0

    def test_finite_round_nan_becomes_zero(self):
        from src.api import _finite_round
        assert _finite_round(float("nan"), 2) == 0.0

    # -- _ndarray_to_b64_png: normalization arithmetic and degenerate branch
    #    (22, 23, 25, 29) --
    def test_normalization_with_nonzero_min_is_pinned(self):
        """Kills 23 (`arr - mn` -> `arr + mn`) and 25 (`mx - mn` -> `mx + mn`):
        the existing regression test uses mn=0, where `arr - mn == arr + mn`
        and `mx - mn == mx + mn` coincide. A non-zero mn distinguishes them."""
        from src.api import _ndarray_to_b64_png
        from PIL import Image
        import base64

        arr = np.array([[2.0, 5.0], [10.0, 20.0]])
        decoded = np.array(Image.open(io.BytesIO(base64.b64decode(_ndarray_to_b64_png(arr)))))
        mn, mx = 2.0, 20.0
        expected = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        assert np.array_equal(decoded, expected)
        assert not np.array_equal(decoded, np.zeros_like(expected))

    def test_constant_float_array_normalizes_to_all_zero_image(self):
        """Kills 22 (`mx > mn` -> `mx >= mn`, which would divide by zero on a
        constant array) and 29 (`np.zeros_like(...)` -> `None`, which would
        crash `Image.fromarray`). A constant non-uint8 array must produce a
        clean all-zero uint8 image via the `else` branch."""
        from src.api import _ndarray_to_b64_png
        from PIL import Image
        import base64

        arr = np.full((4, 4), 3.5, dtype=np.float64)
        decoded = np.array(Image.open(io.BytesIO(base64.b64decode(_ndarray_to_b64_png(arr)))))
        assert np.array_equal(decoded, np.zeros((4, 4), dtype=np.uint8))

    # -- _create_overlay: alpha default, red color, blend formula, final
    #    scale/clip/dtype (33, 39, 40, 41, 44-53) --
    def test_create_overlay_pinned_output_values(self):
        """Pin the exact uint8 RGB values of `_create_overlay` for a tiny
        2x2 image with a known mask. Distinguishes the default alpha (33),
        the red color components (39-41), every blend-formula operator/sign
        mutation (44-49), and the final `* 255 / clip / astype` chain
        (50-53) — a wrong constant anywhere shifts at least one pinned value."""
        from src.api import _create_overlay

        image_float = np.array([[0.0, 1.0], [0.5, 0.2]], dtype=np.float32)
        mask = np.array([[True, False], [False, True]])
        overlay = _create_overlay(image_float, mask)

        assert overlay.dtype == np.uint8
        assert overlay.shape == (2, 2, 3)
        np.testing.assert_array_equal(overlay[0, 0], [102, 0, 0])
        np.testing.assert_array_equal(overlay[0, 1], [255, 255, 255])
        np.testing.assert_array_equal(overlay[1, 0], [127, 127, 127])
        np.testing.assert_array_equal(overlay[1, 1], [132, 30, 30])

    def test_create_overlay_clips_high_values_at_255_not_256(self):
        """Kill mutant 53: `clip(0, 256)` vs `clip(0, 255)`.

        image_float=1.5 at a masked pixel → blended R = 0.6*1.5 + 0.4 = 1.3
        → *255 = 331.5 → clip(0,255) saturates to 255; clip(0,256) allows
        256 → uint8 wraps to 0. Assert R == 255 to distinguish the two."""
        from src.api import _create_overlay

        overlay = _create_overlay(
            np.array([[1.5]], dtype=np.float32),
            np.array([[True]]),
            alpha=0.4,
        )
        assert overlay[0, 0, 0] == 255

    # -- create_app: model-injection gate (55, 56) --
    def test_create_app_without_injected_model_builds_a_working_one(self, tiny_png_bytes):
        """Kills 55 (`model is None` -> `model is not None`) and 56
        (`_model = build_model()` -> `_model = None`): both mutants leave
        `_model` as `None` when no model is injected, which crashes
        `predict(_model, ...)`. Calling create_app() with no model must
        still produce a working /segment endpoint."""
        from src.api import create_app

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200

    # -- /segment validate_file (68, 71) --
    def test_image_with_empty_filename_returns_400(self, client):
        """Kills mutant 68 (`filename == ""` -> `filename == "XXXX"`):
        Flask populates request.files["image"] with filename="" when no
        file is actually selected in the form input."""
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        assert r.get_json() == {"error": "No image file provided"}

    # -- /segment methods fallback (77, 78) --
    def test_no_method_field_falls_back_to_none(self, client, tiny_png_bytes):
        """Kills 77 (`"none"` -> `"XXnoneXX"`, which would fail the
        _SUPPORTED_METHODS check and return 400) and 78
        (`methods = [...]` -> `methods = None`, which crashes the
        `for m in methods` loop): omitting the "method" field entirely
        must default to method="none" and succeed."""
        r = client.post(
            "/segment",
            data={"image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        result = data[0] if isinstance(data, list) else data
        assert result["method"] == "none"

    # -- /segment validate_methods error message (81) --
    def test_unknown_method_error_message_pinned(self, client, tiny_png_bytes):
        r = client.post(
            "/segment",
            data={"method": "bogus", "image": (io.BytesIO(tiny_png_bytes), "t.png")},
            content_type="multipart/form-data",
        )
        assert r.get_json() == {"error": "Unknown despeckle method: 'bogus'"}

    # -- /segment image-decode exception handler (88, 89, 90) --
    def test_unreadable_image_error_response_pinned(self, client):
        """Kills 88 (`"error"` -> `"XXerrorXX"` JSON key), 89 (XX-wrapped
        message), 90 (status 400 -> 401)."""
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(b"not an image"), "bad.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data
        assert data["error"].startswith("Cannot read image: ")

    # -- /segment img_float normalization (91, 92) --
    def test_image_normalized_by_dividing_by_255(self, tiny_png_bytes):
        """Kills 91 (`/ 255.0` -> `* 255.0`) and 92 (`/ 255.0` -> `/ 256.0`):
        spy on the model to capture the (padded) tensor fed to `predict`,
        whose max equals `image.max() / 255.0` exactly when no further
        in-`predict` rescale is triggered (max <= 1.0)."""
        import src.api as api_mod
        import torch
        from PIL import Image

        captured = {}

        class _SpyModel:
            def __call__(self, x):
                captured.setdefault("max", float(x.max()))
                return torch.full_like(x, 0.8)

        app = api_mod.create_app(model=_SpyModel())
        app.config["TESTING"] = True
        client = app.test_client()
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200

        image_arr = np.array(Image.open(io.BytesIO(tiny_png_bytes)).convert("L"), dtype=np.uint8)
        expected_max = float(image_arr.max()) / 255.0
        assert captured["max"] == pytest.approx(expected_max, abs=1e-6)

    # -- /segment denoised_uint8 conversion (109, 110, 111, 112) --
    def test_denoised_uint8_conversion_pinned(self, monkeypatch):
        """Pin `(denoised_float * 255).clip(0, 255).astype(np.uint8)` for
        values straddling both clip bounds and the 1.0 boundary — kills
        109 (`* 255` -> `/ 255`), 110 (`* 256`), 111 (`clip(1, 255)`),
        112 (`clip(0, 256)`, which wraps 256 -> 0 via uint8 overflow)."""
        import src.api as api_mod
        import torch
        from PIL import Image
        import base64

        fixed_row = [-0.1, 0.0, 0.5, 1.05, -0.1, 0.0, 0.5, 1.05]
        fixed_denoised = np.array([fixed_row] * 8, dtype=np.float32)
        monkeypatch.setattr(api_mod, "_despeckle", lambda img, method=None, **kw: fixed_denoised)

        class _M:
            def __call__(self, x):
                return torch.full_like(x, 0.8)

        img4 = Image.fromarray(np.zeros((8, 8), dtype=np.uint8))
        buf = io.BytesIO()
        img4.save(buf, format="PNG")

        app = api_mod.create_app(model=_M())
        app.config["TESTING"] = True
        client = app.test_client()
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(buf.getvalue()), "t.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        result = data[0] if isinstance(data, list) else data

        decoded = np.array(Image.open(io.BytesIO(base64.b64decode(result["denoised"]))))
        expected = np.array([[0, 0, 127, 255, 0, 0, 127, 255]] * 8, dtype=np.uint8)
        assert np.array_equal(decoded, expected)

    # -- /segment metric rounding precision (116, 118, 120, 122, 124, 126) --
    def test_metric_rounding_precision_pinned(self, monkeypatch, tiny_png_bytes):
        """Pin the exact rounding ndigits for every metric in the response
        by stubbing each metric function with a many-decimal-digit constant
        that rounds differently at adjacent precisions — kills 116 (dice
        4->5), 118 (iou 4->5), 120 (hd95 2->3), 122 (psnr ndigits 2->3),
        124 (ssim 4->5), 126 (niqe 4->5)."""
        import src.api as api_mod
        import torch

        monkeypatch.setattr(api_mod, "_dice", lambda *a, **k: 0.123456789)
        monkeypatch.setattr(api_mod, "_iou", lambda *a, **k: 0.234567891)
        monkeypatch.setattr(api_mod, "_hd95", lambda *a, **k: 12.345678)
        monkeypatch.setattr(api_mod, "_psnr", lambda *a, **k: 34.567891)
        monkeypatch.setattr(api_mod, "_ssim", lambda *a, **k: 0.345678912)
        monkeypatch.setattr(api_mod, "_niqe", lambda *a, **k: 0.456789123)

        class _M:
            def __call__(self, x):
                return torch.full_like(x, 0.8)

        app = api_mod.create_app(model=_M())
        app.config["TESTING"] = True
        client = app.test_client()
        r = client.post(
            "/segment",
            data={"method": "none", "image": (io.BytesIO(tiny_png_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        data = r.get_json()
        result = data[0] if isinstance(data, list) else data

        assert result["dice"] == round(0.123456789, 4)
        assert result["iou"] == round(0.234567891, 4)
        assert result["hd95"] == round(12.345678, 2)
        assert result["psnr"] == round(34.567891, 2)
        assert result["ssim"] == round(0.345678912, 4)
        assert result["niqe"] == round(0.456789123, 4)

    # -- error handlers: registration + pinned JSON bodies (141-149) --
    def test_404_handler_returns_pinned_json(self, client):
        """Kills 141 (`@app.errorhandler(404)` -> `(405)`, which silently
        steals the 404 slot via the later 405 registration), 142 (decorator
        removed entirely), 143 (`"error"` -> `"XXerrorXX"`), 144 (XX-wrapped
        "Not found")."""
        r = client.get("/totally-bogus-route-xyz")
        assert r.status_code == 404
        assert r.get_json() == {"error": "Not found"}

    def test_405_handler_returns_pinned_json(self, client):
        """Kills 146 (`@app.errorhandler(405)` -> `(406)`), 147 (decorator
        removed), 148 (`"error"` -> `"XXerrorXX"`), 149 (XX-wrapped
        "Method not allowed")."""
        r = client.get("/segment")
        assert r.status_code == 405
        assert r.get_json() == {"error": "Method not allowed"}

    # -- __main__ guard (151, 152, 153, 154, 155, 156) --
    def test_main_guard_source_is_pinned_verbatim(self):
        """Source-level pin for the `__main__` guard (mutants 151-156).

        We deliberately READ THE FILE rather than importing src.api.
        Importing with mutant 151 applied (`__name__ != "__main__"`)
        causes the guard to fire at import time (since "src.api" !=
        "__main__" is True), immediately calling app.run() and spawning
        a real Werkzeug dev server that is hard to kill and outlives
        the test process. Reading the file as text avoids execution
        entirely, kills the same set of mutants by asserting the exact
        verbatim guard text, and has no side effects whatsoever."""
        from pathlib import Path
        src = Path(__file__).parent.parent / "src" / "api.py"
        source = src.read_text()
        assert (
            'if __name__ == "__main__":\n'
            "    app = create_app()\n"
            "    app.run(debug=True, port=5000, threaded=True)"
        ) in source
