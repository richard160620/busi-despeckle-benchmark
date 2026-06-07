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
    def test_all_despeckle_methods_return_200(self, client, tiny_png_bytes, method):
        """ISP C3: each despeckle method → 200 (Graph Coverage W7: each dispatch edge)."""
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
