"""The insights, aspect and storefront endpoints.

Over the same doubles the rest of the route tests use, so no artifact is
needed. The fake model answers "positive" for everything, which makes the
labels uninteresting and the *shapes* -- status codes, validation, caching,
the 503 and 404 paths -- exactly what these assert.
"""

from __future__ import annotations

import pytest

from app.services import catalogue
from app.services.aspects import load_lexicon

PREFIX = "/api/v1"

REVIEWS = [
    "Battery life is amazing but the screen is dim.",
    "The screen is gorgeous and the battery lasts all day.",
    "The keyboard feels great, but the trackpad is unresponsive.",
]


@pytest.fixture(autouse=True)
def clean_caches():
    """The catalogue and lexicon caches are process-wide, like the registry.

    Cleared around every test so one test's warmed breakdowns cannot make
    another test pass without computing anything -- or fail because a
    previous test cached against a different fake.
    """
    catalogue.load_catalogue.cache_clear()
    catalogue.breakdowns.clear()
    load_lexicon.cache_clear()
    yield
    catalogue.load_catalogue.cache_clear()
    catalogue.breakdowns.clear()
    load_lexicon.cache_clear()


# ---- POST /insights ---------------------------------------------------- #


class TestInsights:
    def test_breaks_a_review_set_down_by_aspect(self, client):
        response = client.post(f"{PREFIX}/insights", json={"reviews": REVIEWS})
        assert response.status_code == 200
        body = response.json()
        assert body["reviews"] == 3
        assert body["sentences"] >= 3
        assert body["predictions"] > 0
        assert body["aspects"], "the laptop vocabulary should match these reviews"
        assert body["model_version"]

    def test_every_aspect_reports_where_it_came_from(self, client):
        # Never from the model: a client has to be able to say so.
        body = client.post(f"{PREFIX}/insights", json={"reviews": REVIEWS}).json()
        for aspect in body["aspects"]:
            assert aspect["source"] in {"corpus", "corpus-rare", "given"}

    def test_shares_sum_to_one_for_every_aspect(self, client):
        body = client.post(f"{PREFIX}/insights", json={"reviews": REVIEWS}).json()
        for aspect in body["aspects"]:
            assert sum(aspect["shares"].values()) == pytest.approx(1.0, abs=1e-3)
            assert sum(aspect["counts"].values()) == aspect["total"]

    def test_every_example_quotes_a_sentence_containing_its_surface(self, client):
        body = client.post(f"{PREFIX}/insights", json={"reviews": REVIEWS}).json()
        seen = 0
        for aspect in body["aspects"]:
            for example in aspect["examples"]:
                assert example["surface"] in example["sentence"]
                assert 0 <= example["review_index"] < len(REVIEWS)
                seen += 1
        assert seen > 0

    def test_judges_caller_supplied_aspects(self, client):
        body = client.post(
            f"{PREFIX}/insights",
            json={"reviews": REVIEWS, "aspects": ["screen"]},
        ).json()
        assert [a["aspect"] for a in body["aspects"]] == ["screen"]
        assert body["aspects"][0]["source"] == "given"

    def test_restricts_extraction_by_domain(self, client):
        body = client.post(
            f"{PREFIX}/insights",
            json={"reviews": ["The food was great."], "domain": "laptop"},
        ).json()
        # "food" is a restaurant aspect, so the laptop vocabulary finds none.
        assert body["aspects"] == []

    def test_finding_no_aspects_is_a_200_not_an_error(self, client):
        # The stated limitation, exercised against the real vocabulary: a
        # review of something SemEval never annotated yields nothing, and
        # that is an answer rather than a failure. ("noise cancelling" would
        # not do here -- "noise" is itself a gold laptop aspect.)
        response = client.post(
            f"{PREFIX}/insights",
            json={"reviews": ["Pollination rates among the hives were excellent."]},
        )
        assert response.status_code == 200
        assert response.json()["aspects"] == []
        assert response.json()["predictions"] == 0

    def test_honours_max_aspects(self, client):
        body = client.post(
            f"{PREFIX}/insights", json={"reviews": REVIEWS, "max_aspects": 2}
        ).json()
        assert len(body["aspects"]) <= 2

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"reviews": []},
            {"reviews": ["   "]},
            {"reviews": [""]},
            {"reviews": ["ok"], "max_aspects": 0},
            {"reviews": ["ok"], "max_aspects": 99},
            {"reviews": ["x" * 5001]},
            {"reviews": ["ok"] * 201},
        ],
    )
    def test_rejects_a_malformed_body_with_422(self, client, payload):
        assert client.post(f"{PREFIX}/insights", json=payload).status_code == 422

    def test_deduplicates_and_trims_supplied_aspects(self, client):
        body = client.post(
            f"{PREFIX}/insights",
            json={"reviews": REVIEWS, "aspects": ["screen", " SCREEN ", "screen"]},
        ).json()
        assert len(body["aspects"]) == 1

    def test_answers_503_when_no_model_is_loaded(self, degraded_client):
        response = degraded_client.post(f"{PREFIX}/insights", json={"reviews": REVIEWS})
        assert response.status_code == 503
        assert "error" in response.json()

    def test_a_503_does_not_leak_the_load_failure(self, degraded_client):
        body = degraded_client.post(f"{PREFIX}/insights", json={"reviews": REVIEWS}).json()
        assert "artifacts" not in str(body).lower()
        assert "traceback" not in str(body).lower()


# ---- GET /aspects ------------------------------------------------------ #


class TestCandidateAspects:
    def test_names_the_aspects_in_a_text(self, client):
        response = client.get(
            f"{PREFIX}/aspects",
            params={"text": "Battery life is amazing but the screen is dim."},
        )
        assert response.status_code == 200
        assert "screen" in [a.lower() for a in response.json()]

    def test_needs_no_model_at_all(self, degraded_client):
        # Pure string matching, so a UI can show the chips while the artifact
        # is still loading.
        response = degraded_client.get(
            f"{PREFIX}/aspects", params={"text": "The screen is dim."}
        )
        assert response.status_code == 200
        assert response.json()

    def test_rejects_blank_text(self, client):
        assert client.get(f"{PREFIX}/aspects", params={"text": "   "}).status_code == 422

    def test_requires_text(self, client):
        assert client.get(f"{PREFIX}/aspects").status_code == 422

    def test_honours_the_limit(self, client):
        response = client.get(
            f"{PREFIX}/aspects",
            params={"text": "screen keyboard battery price food service", "limit": 2},
        )
        assert len(response.json()) <= 2

    @pytest.mark.parametrize("limit", [0, 25])
    def test_rejects_an_out_of_range_limit(self, client, limit):
        response = client.get(
            f"{PREFIX}/aspects", params={"text": "The screen is dim.", "limit": limit}
        )
        assert response.status_code == 422


# ---- the storefront ---------------------------------------------------- #


class TestCatalogue:
    def test_lists_the_products_without_predicting(self, degraded_client):
        # The shelf has to render before any model work, so this endpoint
        # must work with no model at all.
        response = degraded_client.get(f"{PREFIX}/products")
        assert response.status_code == 200
        products = response.json()
        assert len(products) >= 4
        for product in products:
            assert product["category"] in {"laptop", "restaurant"}
            assert product["review_count"] > 0
            assert product["slug"] and product["name"]

    def test_a_product_page_carries_its_reviews_and_breakdown(self, client):
        slug = client.get(f"{PREFIX}/products").json()[0]["slug"]
        response = client.get(f"{PREFIX}/products/{slug}")
        assert response.status_code == 200
        body = response.json()
        assert body["slug"] == slug
        assert len(body["reviews"]) == body["review_count"]
        assert body["insights"]["aspects"]
        assert body["insights"]["predictions"] > 0

    def test_a_product_s_aspects_come_from_its_category(self, client):
        products = client.get(f"{PREFIX}/products").json()
        laptop = next(p for p in products if p["category"] == "laptop")
        body = client.get(f"{PREFIX}/products/{laptop['slug']}").json()
        found = {a["aspect"] for a in body["insights"]["aspects"]}
        # Fixed per category, so the shelf can be sorted by one of them.
        assert found <= {"battery life", "screen", "keyboard", "price", "performance"}

    def test_unknown_product_is_404_in_the_standard_error_shape(self, client):
        response = client.get(f"{PREFIX}/products/not-a-product")
        assert response.status_code == 404
        assert set(response.json()) == {"error", "status"}

    def test_the_insights_path_is_not_read_as_a_slug(self, client):
        # /products/insights is declared before /products/{slug}; if that
        # ever changed, this would 404 instead.
        assert client.get(f"{PREFIX}/products/insights").status_code == 200

    def test_catalogue_insights_scores_every_product(self, client):
        body = client.get(f"{PREFIX}/products/insights").json()
        slugs = {p["slug"] for p in client.get(f"{PREFIX}/products").json()}
        assert set(body["products"]) == slugs
        assert body["aspects"]
        for scores in body["products"].values():
            for score in scores:
                assert 0.0 <= score["positive_share"] <= 1.0
                assert 0.0 <= score["negative_share"] <= 1.0
                assert score["total"] > 0

    def test_a_breakdown_is_computed_once_per_product(self, client, fake_model):
        slug = client.get(f"{PREFIX}/products").json()[0]["slug"]
        client.get(f"{PREFIX}/products/{slug}")
        after_first = len(fake_model.calls)
        assert after_first > 0

        client.get(f"{PREFIX}/products/{slug}")
        assert len(fake_model.calls) == after_first, "second request should be cached"

    def test_predictions_are_batched_rather_than_one_per_pair(self, client, fake_model):
        slug = client.get(f"{PREFIX}/products").json()[0]["slug"]
        body = client.get(f"{PREFIX}/products/{slug}").json()
        sizes = fake_model.batch_sizes()
        assert sizes, "no batched forward pass was made"
        # One pass per chunk, not one per pair.
        assert len(sizes) < body["insights"]["predictions"]
        assert max(sizes) > 1

    def test_a_product_page_answers_503_with_no_model(self, degraded_client):
        slug = degraded_client.get(f"{PREFIX}/products").json()[0]["slug"]
        assert degraded_client.get(f"{PREFIX}/products/{slug}").status_code == 503

    def test_catalogue_insights_answers_503_with_no_model(self, degraded_client):
        assert degraded_client.get(f"{PREFIX}/products/insights").status_code == 503


class TestCatalogueData:
    """The committed data file, which ships where the corpus does not."""

    def test_the_catalogue_is_present_and_coherent(self):
        catalogue.load_catalogue.cache_clear()
        products, aspects = catalogue.load_catalogue()
        assert len(products) >= 4
        assert set(aspects) >= {"laptop", "restaurant"}

        slugs = [product.slug for product in products]
        assert len(slugs) == len(set(slugs)), "slugs must be unique"

        for product in products:
            assert product.category in aspects
            assert product.reviews
            assert product.provenance, "each product states how much is held out"

    def test_no_review_sentence_is_shared_between_products(self):
        catalogue.load_catalogue.cache_clear()
        products, _ = catalogue.load_catalogue()
        seen: set[str] = set()
        for product in products:
            for review in product.reviews:
                assert review not in seen, "two products share evidence"
                seen.add(review)

    def test_an_absent_catalogue_is_empty_not_fatal(self, tmp_path):
        catalogue.load_catalogue.cache_clear()
        try:
            assert catalogue.load_catalogue(str(tmp_path / "nope.json")) == ((), {})
        finally:
            catalogue.load_catalogue.cache_clear()
