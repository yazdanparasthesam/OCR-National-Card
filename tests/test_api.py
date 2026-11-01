"""HTTP contract of the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nidocr.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """An HTTP client bound to a freshly created FastAPI app."""
    return TestClient(create_app())


def test_health(client):
    """``/health`` reports the engine as ready and lists the readable fields."""
    payload = client.get("/api/v1/health").json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["version"]
    assert set(payload["fields"]) >= {"national_code", "first_name", "last_name"}


def test_index_lists_endpoints(client):
    """The landing page advertises the version and the main endpoints."""
    payload = client.get("/").json()
    assert payload["version"]
    assert "card" in payload["endpoints"]


def test_field_list_and_metadata(client):
    """``/fields`` lists every field and ``/fields/{key}`` returns its metadata (404 otherwise)."""
    listed = client.get("/api/v1/fields").json()
    assert {item["key"] for item in listed} >= {"national_code", "birth_date"}
    meta = client.get("/api/v1/fields/national_code").json()
    assert meta["path"] == "national-code"
    assert client.get("/api/v1/fields/nope").status_code == 404


def test_templates(client):
    """``/templates`` lists the shipped templates and 404s on an unknown name."""
    templates = client.get("/api/v1/templates").json()
    names = {item["name"] for item in templates}
    assert "national_card_front" in names
    assert client.get("/api/v1/templates/national_card_front").status_code == 200
    assert client.get("/api/v1/templates/missing").status_code == 404


def test_card_upload(client, sample_card_file):
    """``POST /card`` reads a whole card and the national code matches the ground truth."""
    path, fields = sample_card_file
    with path.open("rb") as handle:
        response = client.post("/api/v1/card", files={"file": ("card.png", handle.read(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["fields"]["national_code"]["value"] == fields["national_code"]
    assert payload["field_order"][0] == "national_code"


def test_single_field_route(client, sample_card_file):
    """``POST /fields/{key}`` returns only the requested field."""
    path, fields = sample_card_file
    with path.open("rb") as handle:
        response = client.post(
            "/api/v1/fields/last_name",
            files={"file": ("card.png", handle.read(), "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["field"]["key"] == "last_name"


def test_unknown_field_route(client, sample_card_file):
    """An unknown field key is answered with 404, not a crash."""
    path, _ = sample_card_file
    with path.open("rb") as handle:
        response = client.post("/api/v1/fields/unknown", files={"file": ("c.png", handle.read(), "image/png")})
    assert response.status_code == 404


def test_rejects_non_image_upload(client):
    """A text upload is rejected with 415/422 instead of reaching the OCR."""
    response = client.post("/api/v1/card", files={"file": ("a.txt", b"not an image", "text/plain")})
    assert response.status_code in {415, 422}


def test_rejects_empty_upload(client):
    """An empty upload is rejected with 422."""
    response = client.post("/api/v1/card", files={"file": ("a.png", b"", "image/png")})
    assert response.status_code == 422


def test_batch_route(client, sample_card_file):
    """``POST /card/batch`` returns one result per uploaded file."""
    path, _ = sample_card_file
    payload = path.read_bytes()
    response = client.post(
        "/api/v1/card/batch",
        files=[("files", ("a.png", payload, "image/png")), ("files", ("b.png", payload, "image/png"))],
    )
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_standalone_validators(client):
    """The image-free validators return the documented repair/validity flags."""
    valid = client.post("/api/v1/validate/national-code", json={"value": "1344005985"}).json()
    assert valid["valid"] is True
    repaired = client.post("/api/v1/validate/national-code", json={"value": "1344005989"}).json()
    assert repaired["valid"] is True and repaired["repaired"] is True
    invalid = client.post("/api/v1/validate/national-code", json={"value": "0000000000"}).json()
    assert invalid["valid"] is False
    date = client.post("/api/v1/validate/birth-date", json={"value": "1360/01/01"}).json()
    assert date["valid"] is True


def test_openapi_schema_is_valid(client):
    """The generated OpenAPI document exposes the per-field route."""
    schema = client.get("/openapi.json").json()
    assert "paths" in schema
    assert "/api/v1/fields/{field}" in schema["paths"]
