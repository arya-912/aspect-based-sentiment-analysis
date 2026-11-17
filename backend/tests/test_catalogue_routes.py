"""The catalogue upload, status, sample and reset endpoints.

Over the fake model, so no artifact is needed. What these pin is the
contract around replacing the active catalogue: the upload takes effect
immediately, the cached scores go with it, the demo catalogue is the default
and is returned to on reset, and every rejection is a 422 whose message says
what to fix.
"""

from __future__ import annotations

import time

import pytest

from app.services import catalogue
from app.services.aspects import load_lexicon

PREFIX = "/api/v1"

#: "Aurora 14" is deliberately also a demo product and "Harbour Grill"
#: deliberately is not: the first proves an upload shadows a demo product of
#: the same name, the second proves reset really drops the uploaded ones.
GOOD_CSV = b"""product_name,category,price,review_text,rating
Aurora 14,laptop,$1099,Battery life is amazing but the screen is dim.,4
Aurora 14,laptop,$1099,The keyboard feels great for long typing sessions.,5
Aurora 14,laptop,$1099,Screen is washed out in sunlight unfortunately.,2
Harbour Grill,restaurant,$$$,Great food but the service was dreadful.,3
Harbour Grill,restaurant,$$$,The atmosphere is wonderful and the prices are fair.,5
"""


def upload(client, data: bytes, filename: str = "catalogue.csv", category: str | None = None):
    files = {"file": (filename, data, "text/csv")}
    form = {"default_category": category} if category else None
    return client.post(f"{PREFIX}/catalogue/upload", files=files, data=form)


def wait_for_scoring(client, timeout: float = 10.0) -> dict:
    """Block until the background pass finishes, so assertions are stable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"{PREFIX}/catalogue/status").json()
        if not status["scoring"] and status["scored"] >= status["products"]:
            return status
        time.sleep(0.02)
    raise AssertionError(f"scoring did not finish within {timeout}s")


@pytest.fixture(autouse=True)
def clean_state():
    """The active catalogue and the breakdowns are process-wide."""
    # Wait out any pass still running from a previous test: the worker is a
    # daemon thread, and a leaked one makes the next test's scoring guard
    # refuse to start.
    catalogue.breakdowns.wait_for_idle()
    catalogue.load_catalogue.cache_clear()
    catalogue.active.reset()
    catalogue.breakdowns.clear()
    load_lexicon.cache_clear()
    yield
    catalogue.breakdowns.wait_for_idle()
    catalogue.load_catalogue.cache_clear()
    catalogue.active.reset()
    catalogue.breakdowns.clear()
    load_lexicon.cache_clear()


# ---- the default ------------------------------------------------------- #


class TestTheDemoCatalogueIsTheDefault:
    def test_status_reports_the_demo_before_any_upload(self, client):
        status = client.get(f"{PREFIX}/catalogue/status").json()
        assert status["source"] == "demo"
        assert "demo" in status["source_name"].lower()
        assert status["products"] >= 4

    def test_status_works_with_no_model_loaded(self, degraded_client):
        # The page must be able to say which file it is showing before the
        # artifact finishes loading.
        response = degraded_client.get(f"{PREFIX}/catalogue/status")
        assert response.status_code == 200
        assert response.json()["source"] == "demo"

    def test_the_shelf_is_the_demo_catalogue(self, client):
        names = {p["name"] for p in client.get(f"{PREFIX}/products").json()}
        assert "Aurora 14" in names


# ---- uploading --------------------------------------------------------- #


class TestUpload:
    def test_replaces_the_active_catalogue(self, client):
        response = upload(client, GOOD_CSV)
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "upload"
        assert body["source_name"] == "catalogue.csv"
        assert body["products"] == 2

        shelf = client.get(f"{PREFIX}/products").json()
        assert {p["name"] for p in shelf} == {"Aurora 14", "Harbour Grill"}

    def test_reports_what_it_kept(self, client):
        body = upload(client, GOOD_CSV).json()
        assert body["rows_used"] == 5
        assert body["products_found"] == 2
        assert body["truncated_products"] is False
        assert body["truncated_reviews"] is False

    def test_carries_the_star_rating_onto_the_shelf(self, client):
        upload(client, GOOD_CSV)
        shelf = {p["name"]: p for p in client.get(f"{PREFIX}/products").json()}
        aurora = shelf["Aurora 14"]
        assert aurora["rating"] == pytest.approx(3.67, abs=0.01)
        assert aurora["rating_count"] == 3

    def test_says_the_numbers_are_not_the_demo_s(self, client):
        # The honesty note the UI prints verbatim.
        body = upload(client, GOOD_CSV).json()
        assert "uploaded" in body["note"].lower()

    def test_a_file_with_no_category_column_needs_one_choosing(self, client):
        data = b"name,text\nTurning Point,If you eat here be aware it takes a while.\n"
        rejected = upload(client, data)
        assert rejected.status_code == 422

        accepted = upload(client, data, category="restaurant")
        assert accepted.status_code == 200
        assert accepted.json()["products"] == 1

    def test_accepts_the_reference_dataset_s_headers(self, client):
        data = (
            b"Sl No,stars_review,text,date,name,categories,city,state\n"
            b"1,3,If you decide to eat here just be aware it will take a while.,"
            b"07-07-18,Turning Point,Restaurants,North Wales,PA\n"
        )
        response = upload(client, data, filename="yelp.csv", category="restaurant")
        assert response.status_code == 200
        assert response.json()["source_name"] == "yelp.csv"

    def test_an_upload_clears_the_previous_catalogue_s_cached_scores(self, client):
        # Score the demo catalogue first.
        client.get(f"{PREFIX}/products/insights")
        wait_for_scoring(client)
        assert client.get(f"{PREFIX}/catalogue/status").json()["scored"] > 0

        upload(client, GOOD_CSV)
        after = client.get(f"{PREFIX}/catalogue/status").json()
        # The new catalogue's own scoring may already have started, but it
        # can never report more scored products than it has.
        assert after["products"] == 2
        assert after["scored"] <= 2

    def test_products_from_an_upload_can_be_opened(self, client):
        upload(client, GOOD_CSV)
        slug = client.get(f"{PREFIX}/products").json()[0]["slug"]
        body = client.get(f"{PREFIX}/products/{slug}").json()
        assert body["insights"]["predictions"] > 0
        assert body["reviews"]

    def test_a_demo_product_is_gone_after_an_upload_replaces_it(self, client):
        upload(client, GOOD_CSV)
        assert client.get(f"{PREFIX}/products/saffron-lane").status_code == 404

    def test_warns_when_the_caps_bit(self, client):
        rows = [b"product_name,category,review_text"]
        for index in range(40):
            rows.append(b"P%d,laptop,The screen is dim indeed here number one.n" % index)
        response = upload(client, b"\n".join(rows) + b"\n")
        warnings = " ".join(response.json()["warnings"]).lower()
        assert "most-reviewed" in warnings

    def test_warns_when_the_file_has_no_ratings(self, client):
        data = b"product_name,category,review_text\nAurora,laptop,The screen is dim indeed.\n"
        warnings = " ".join(upload(client, data).json()["warnings"]).lower()
        assert "star rating" in warnings

    def test_an_upload_needs_no_model(self, degraded_client):
        # Parsing is CSV work; only the scoring needs an artifact.
        response = upload(degraded_client, GOOD_CSV)
        assert response.status_code == 200
        assert response.json()["products"] == 2


class TestUploadRejections:
    def test_requires_a_file(self, client):
        assert client.post(f"{PREFIX}/catalogue/upload").status_code == 422

    def test_rejects_an_empty_file(self, client):
        response = upload(client, b"")
        assert response.status_code == 422
        assert "empty" in response.json()["error"].lower()

    def test_rejects_missing_columns_and_names_them(self, client):
        response = upload(client, b"foo,bar\n1,2\n")
        assert response.status_code == 422
        message = response.json()["error"]
        assert "product_name" in message and "review_text" in message

    def test_rejects_an_unsupported_category(self, client):
        data = b"product_name,category,review_text\nPixel,phone,The camera is superb here.\n"
        response = upload(client, data)
        assert response.status_code == 422
        # The message states the real reason: the vocabulary's coverage.
        assert "laptop" in response.json()["error"]

    def test_rejects_a_file_that_is_not_utf8(self, client):
        data = b"product_name,category,review_text\n\xff\xfe,laptop,The screen is dim.\n"
        response = upload(client, data)
        assert response.status_code == 422
        assert "UTF-8" in response.json()["error"]

    def test_rejects_a_file_with_no_usable_rows(self, client):
        data = b"product_name,category,review_text\n,laptop,short\n"
        response = upload(client, data)
        assert response.status_code == 422
        assert "no usable rows" in response.json()["error"]

    def test_a_rejection_leaves_the_previous_catalogue_serving(self, client):
        upload(client, GOOD_CSV)
        assert client.get(f"{PREFIX}/catalogue/status").json()["products"] == 2

        assert upload(client, b"foo,bar\n1,2\n").status_code == 422
        # Still the good upload, not the demo and not an empty shelf.
        status = client.get(f"{PREFIX}/catalogue/status").json()
        assert status["products"] == 2
        assert status["source_name"] == "catalogue.csv"

    def test_every_rejection_uses_the_standard_error_shape(self, client):
        response = upload(client, b"")
        assert set(response.json()) == {"error", "status"}


# ---- reset ------------------------------------------------------------- #


class TestReset:
    def test_goes_back_to_the_demo_catalogue(self, client):
        upload(client, GOOD_CSV)
        assert client.get(f"{PREFIX}/catalogue/status").json()["source"] == "upload"

        body = client.post(f"{PREFIX}/catalogue/reset").json()
        assert body["source"] == "demo"
        names = {p["name"] for p in client.get(f"{PREFIX}/products").json()}
        assert "Aurora 14" in names
        assert "Harbour Grill" not in names

    def test_is_harmless_when_already_on_the_demo(self, client):
        assert client.post(f"{PREFIX}/catalogue/reset").json()["source"] == "demo"


# ---- the sample file --------------------------------------------------- #


class TestSample:
    def test_is_served_as_a_downloadable_csv(self, client):
        response = client.get(f"{PREFIX}/catalogue/sample")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]

    def test_round_trips_through_the_parser_it_documents(self, client):
        # A sample the loader would reject is worse than none.
        sample = client.get(f"{PREFIX}/catalogue/sample").content
        response = upload(client, sample, filename="absa-catalogue-sample.csv")
        assert response.status_code == 200
        assert response.json()["products"] >= 2

    def test_needs_no_model(self, degraded_client):
        assert degraded_client.get(f"{PREFIX}/catalogue/sample").status_code == 200


# ---- progress ---------------------------------------------------------- #


class TestScoringProgress:
    def test_status_reports_progress_towards_the_total(self, client):
        upload(client, GOOD_CSV)
        status = wait_for_scoring(client)
        assert status["scored"] == status["products"] == 2
        assert status["scoring"] is False

    def test_scores_appear_in_the_catalogue_insights(self, client):
        upload(client, GOOD_CSV)
        wait_for_scoring(client)
        body = client.get(f"{PREFIX}/products/insights").json()
        assert body["scored"] == 2
        assert set(body["products"]) == {
            p["slug"] for p in client.get(f"{PREFIX}/products").json()
        }

    def test_insights_answers_503_with_no_model_rather_than_an_empty_map(
        self, degraded_client
    ):
        upload(degraded_client, GOOD_CSV)
        assert degraded_client.get(f"{PREFIX}/products/insights").status_code == 503
