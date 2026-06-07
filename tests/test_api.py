"""Tests for src/api.py — Flask endpoint tests.

Testing methods applied:
  - Input Space Partitioning (W6): route × method × file presence × despeckle method
  - Graph Coverage (W7): every route + error-handler edge
  - Boundary conditions: missing file, unknown method, wrong HTTP verb

Status codes covered: 200, 400 (missing file), 400 (unknown method), 404, 405.
All tests inject a mock model — no GPU or real model required.
"""
import io
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
