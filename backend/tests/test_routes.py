"""Route, schema and error-handling tests. No artifact required."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import LABELS, FakeModel

PREFIX = "/api/v1"
VALID_SENTIMENTS = set(LABELS)


# ---- health ----------------------------------------------------------- #


def test_health_returns_200(client):
    response = client.get(f"{PREFIX}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_degraded_without_a_model(degraded_client):
    """Still 200: "up but not ready" must be distinguishable from "unreachable"."""
    response = degraded_client.get(f"{PREFIX}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False}


# ---- predict: happy path ---------------------------------------------- #


def test_predict_valid(client):
    response = client.post(
        f"{PREFIX}/predict",
        json={"text": "Battery is excellent.", "aspect": "battery"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["aspect"] == "battery"
    assert body["sentiment"] in VALID_SENTIMENTS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "absa-v1-test"


def test_predict_uses_the_fixed_logits(client):
    """Logits [0.1, 0.2, 0.7] -> positive, softmax 0.4565 (4dp)."""
    response = client.post(
        f"{PREFIX}/predict", json={"text": "Battery is excellent.", "aspect": "battery"}
    )
    body = response.json()
    assert body["sentiment"] == "positive"

    expected = float(torch.softmax(torch.tensor([0.1, 0.2, 0.7]), dim=-1)[2])
    assert body["confidence"] == pytest.approx(round(expected, 4))


def test_predict_response_has_exactly_the_documented_fields(client):
    body = client.post(
        f"{PREFIX}/predict", json={"text": "Great screen.", "aspect": "screen"}
    ).json()
    assert set(body) == {"aspect", "sentiment", "confidence", "model_version"}


def test_predict_passes_the_model_a_batch_of_one(client, fake_model, fake_encoder):
    client.post(f"{PREFIX}/predict", json={"text": "Great screen.", "aspect": "screen"})

    assert fake_encoder.calls == [("Great screen.", "screen")]
    batch = fake_model.calls[-1]
    assert set(batch) == {
        "input_ids", "attention_mask", "token_type_ids", "aspect_mask", "context_mask"
    }
    for name, tensor in batch.items():
        assert tensor.shape[0] == 1, f"{name} should carry a batch dimension"


def test_predict_echoes_the_aspect_verbatim(client):
    body = client.post(
        f"{PREFIX}/predict", json={"text": "The track pad is unresponsive.", "aspect": "track pad"}
    ).json()
    assert body["aspect"] == "track pad"


@pytest.mark.parametrize(
    "winner,expected",
    [(0, "negative"), (1, "neutral"), (2, "positive")],
)
def test_predict_decodes_every_label(client, mocked_registry, winner, expected):
    logits = torch.full((1, 3), -5.0)
    logits[0, winner] = 5.0
    mocked_registry._model = FakeModel(logits)

    body = client.post(
        f"{PREFIX}/predict", json={"text": "Some review text.", "aspect": "thing"}
    ).json()
    assert body["sentiment"] == expected
    assert body["confidence"] > 0.9


# ---- predict: validation ---------------------------------------------- #


def test_predict_empty_text(client):
    response = client.post(f"{PREFIX}/predict", json={"text": "", "aspect": "battery"})
    assert response.status_code == 422
    assert response.json()["status"] == 422
    assert "text" in response.json()["error"]


def test_predict_missing_aspect(client):
    response = client.post(f"{PREFIX}/predict", json={"text": "Great laptop"})
    assert response.status_code == 422
    assert "aspect" in response.json()["error"]


def test_predict_empty_aspect(client):
    response = client.post(f"{PREFIX}/predict", json={"text": "Great laptop", "aspect": ""})
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["text", "aspect"])
def test_predict_rejects_whitespace_only(client, field):
    """`min_length` alone would let " " through to the tokenizer."""
    payload = {"text": "Great laptop", "aspect": "screen"}
    payload[field] = "   "
    response = client.post(f"{PREFIX}/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_text_over_the_limit(client):
    response = client.post(
        f"{PREFIX}/predict", json={"text": "x" * 2001, "aspect": "battery"}
    )
    assert response.status_code == 422


def test_predict_rejects_aspect_over_the_limit(client):
    response = client.post(
        f"{PREFIX}/predict", json={"text": "Fine laptop", "aspect": "y" * 201}
    )
    assert response.status_code == 422


def test_predict_accepts_the_boundary_lengths(client):
    response = client.post(
        f"{PREFIX}/predict", json={"text": "x" * 2000, "aspect": "y" * 200}
    )
    assert response.status_code == 200


def test_predict_rejects_an_empty_body(client):
    assert client.post(f"{PREFIX}/predict", json={}).status_code == 422


def test_predict_rejects_wrong_types(client):
    response = client.post(f"{PREFIX}/predict", json={"text": 42, "aspect": ["a"]})
    assert response.status_code == 422


def test_predict_rejects_malformed_json(client):
    response = client.post(
        f"{PREFIX}/predict",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ---- predict: errors from below --------------------------------------- #


def test_predict_returns_400_when_the_encoder_rejects_the_input(client, mocked_registry):
    """A tokenizer rejection is the caller's problem, so 400 -- not 500."""

    class RejectingEncoder:
        max_length = 128

        def encode(self, text, aspect, label=None):
            raise ValueError(f"aspect produced zero tokens: {aspect!r}")

    mocked_registry._encoder = RejectingEncoder()
    response = client.post(
        f"{PREFIX}/predict", json={"text": "Fine laptop", "aspect": "​"}
    )
    assert response.status_code == 400
    assert response.json()["status"] == 400
    assert "zero tokens" in response.json()["error"]


def test_predict_returns_503_without_a_model(degraded_client):
    response = degraded_client.post(
        f"{PREFIX}/predict", json={"text": "Battery is great.", "aspect": "battery"}
    )
    assert response.status_code == 503
    assert response.json()["status"] == 503
    assert "not loaded" in response.json()["error"]


def test_503_does_not_leak_the_artifact_path(unloaded_registry, settings):
    """The load failure reason is logged, not served -- it can carry paths."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    unloaded_registry._load_error = "FileNotFoundError: /srv/secret/absa-v1"
    with TestClient(create_app(settings)) as leaky_client:
        response = leaky_client.get(f"{PREFIX}/model")
    assert response.status_code == 503
    assert "secret" not in response.text
    assert "/srv" not in response.text


def test_predict_returns_500_and_hides_internals(raising_client, mocked_registry):
    """An unexpected failure must not leak its message to the client."""

    class ExplodingModel:
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA kernel panic at /secret/path/model.pt")

        def parameters(self):
            return iter(())

    mocked_registry._model = ExplodingModel()
    response = raising_client.post(
        f"{PREFIX}/predict", json={"text": "Battery is great.", "aspect": "battery"}
    )
    assert response.status_code == 500
    body = response.json()
    assert body == {"error": "internal server error", "status": 500}
    assert "secret" not in response.text
    assert "Traceback" not in response.text


# ---- model metadata --------------------------------------------------- #


def test_model_endpoint_returns_200(client):
    response = client.get(f"{PREFIX}/model")
    assert response.status_code == 200

    body = response.json()
    assert body["version"] == "absa-v1-test"
    assert body["base_model"] == "bert-base-uncased"
    assert body["task"] == "aspect-based-sentiment-analysis"
    assert body["labels"] == list(LABELS)


def test_model_endpoint_includes_the_evaluation_metrics(client):
    evaluation = client.get(f"{PREFIX}/model").json()["evaluation"]
    assert evaluation["accuracy"] == pytest.approx(0.7821)
    assert evaluation["macro_f1"] == pytest.approx(0.7447)
    assert set(evaluation["per_class"]) == set(LABELS)
    assert len(evaluation["confusion_matrix"]) == 3


def test_model_endpoint_tolerates_missing_metrics(client, mocked_registry, tmp_path):
    """A model can serve predictions without its evaluation report."""
    empty = tmp_path / "no-metrics"
    empty.mkdir()
    mocked_registry._artifact_dir = empty

    response = client.get(f"{PREFIX}/model")
    assert response.status_code == 200
    assert response.json()["evaluation"] is None


def test_model_endpoint_returns_503_without_a_model(degraded_client):
    response = degraded_client.get(f"{PREFIX}/model")
    assert response.status_code == 503
    assert response.json()["status"] == 503


# ---- app wiring ------------------------------------------------------- #


def test_routes_are_mounted_under_the_versioned_prefix(client):
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/health").status_code == 404


def test_root_points_at_the_docs(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert body["health"] == f"{PREFIX}/health"


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert f"{PREFIX}/predict" in schema["paths"]
    assert f"{PREFIX}/health" in schema["paths"]
    assert f"{PREFIX}/model" in schema["paths"]


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://localhost:3000"])
def test_cors_allows_the_dev_servers(client, origin):
    response = client.options(
        f"{PREFIX}/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_rejects_an_unlisted_origin(client):
    response = client.options(
        f"{PREFIX}/predict",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") != "http://evil.example"


def test_unknown_route_is_404(client):
    response = client.get(f"{PREFIX}/nope")
    assert response.status_code == 404
    # The router raises Starlette's HTTPException, not FastAPI's subclass, so
    # this also pins that the handler is registered against the base class.
    assert response.json() == {"error": "Not Found", "status": 404}


def test_wrong_method_is_405(client):
    response = client.get(f"{PREFIX}/predict")
    assert response.status_code == 405
    assert response.json()["status"] == 405


@pytest.mark.parametrize(
    "request_kwargs,expected",
    [
        ({"json": {"text": "", "aspect": "a"}}, 422),
        ({"json": {"text": "ok"}}, 422),
    ],
)
def test_every_error_uses_the_documented_shape(client, request_kwargs, expected):
    response = client.post(f"{PREFIX}/predict", **request_kwargs)
    assert response.status_code == expected
    assert set(response.json()) == {"error", "status"}
    assert response.json()["status"] == expected
