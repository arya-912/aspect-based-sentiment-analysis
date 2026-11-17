"""Parsing an uploaded catalogue CSV.

Pure string and grouping work -- no model, no artifact. The rejections are
the bulk of it: every one of them is something the caller can fix, so each
has to produce a message that names what to fix rather than a generic 422.
"""

from __future__ import annotations

import pytest

from app.services.catalogue_csv import (
    MAX_BYTES,
    MAX_ROWS,
    MIN_REVIEW_CHARS,
    CatalogueCsvError,
    parse_catalogue_csv,
    slugify,
)

CANONICAL = b"""product_name,category,price,review_text,rating
Aurora 14,laptop,$1099,Battery life is amazing but the screen is dim.,4
Aurora 14,laptop,$1099,The keyboard feels great for long sessions.,5
Olive & Ash,restaurant,$$$,Great food but the service was dreadful.,3
"""

ASPECTS = {"laptop": ("screen", "battery life"), "restaurant": ("food", "service")}


def parse(data: bytes, **kwargs):
    kwargs.setdefault("aspects_by_category", ASPECTS)
    return parse_catalogue_csv(data, **kwargs)


class TestTheCanonicalFormat:
    def test_groups_rows_into_products(self):
        parsed = parse(CANONICAL)
        assert [p.name for p in parsed.products] == ["Aurora 14", "Olive & Ash"]
        assert len(parsed.products[0].reviews) == 2
        assert parsed.rows_read == 3
        assert parsed.rows_used == 3
        assert parsed.products_found == 2

    def test_keeps_the_category_per_product(self):
        parsed = parse(CANONICAL)
        assert {p.name: p.category for p in parsed.products} == {
            "Aurora 14": "laptop",
            "Olive & Ash": "restaurant",
        }

    def test_takes_the_price_from_the_first_row_that_has_one(self):
        assert parse(CANONICAL).products[0].price == "$1099"

    def test_averages_the_star_ratings(self):
        parsed = parse(CANONICAL)
        aurora = parsed.products[0]
        assert aurora.rating == 4.5
        assert aurora.rating_count == 2

    def test_reports_no_rating_when_the_column_is_absent(self):
        data = b"product_name,category,review_text\nAurora,laptop,The screen is dim indeed.\n"
        product = parse(data).products[0]
        assert product.rating is None
        assert product.rating_count == 0

    def test_offers_the_sort_aspects_of_the_categories_present(self):
        parsed = parse(CANONICAL)
        assert set(parsed.category_aspects) == {"laptop", "restaurant"}
        assert parsed.category_aspects["laptop"] == ("screen", "battery life")


class TestAliasedHeaders:
    def test_accepts_the_reference_dataset_s_headers(self):
        # The Yelp restaurant extract this was built against: `name`, `text`,
        # `stars_review`, and no category column at all.
        data = (
            b"Sl No,stars_review,text,date,name,categories,city,state\n"
            b"1,3,If you decide to eat here just be aware it will take a while.,"
            b"07-07-18,Turning Point,Restaurants,North Wales,PA\n"
        )
        parsed = parse(data, default_category="restaurant")
        assert [p.name for p in parsed.products] == ["Turning Point"]
        assert parsed.products[0].category == "restaurant"
        assert parsed.products[0].rating == 3.0

    @pytest.mark.parametrize("header", [b"name", b"business_name", b"product", b"title"])
    def test_accepts_each_product_name_alias(self, header):
        data = b"%s,review_text\nAurora,The screen is dim indeed.\n" % header
        assert parse(data, default_category="laptop").products[0].name == "Aurora"

    @pytest.mark.parametrize("header", [b"text", b"review", b"content", b"body"])
    def test_accepts_each_review_text_alias(self, header):
        data = b"product_name,%s\nAurora,The screen is dim indeed.\n" % header
        assert len(parse(data, default_category="laptop").products[0].reviews) == 1

    def test_ignores_header_case_and_a_spreadsheet_bom(self):
        data = "﻿Product_Name,Review_Text\nAurora,The screen is dim indeed.\n".encode()
        assert parse(data, default_category="laptop").products[0].name == "Aurora"


class TestCategory:
    def test_a_default_covers_a_file_with_no_category_column(self):
        data = b"product_name,review_text\nAurora,The screen is dim indeed.\n"
        assert parse(data, default_category="laptop").products[0].category == "laptop"

    def test_the_column_wins_over_the_default(self):
        data = b"product_name,category,review_text\nOlive,restaurant,Great food here truly.\n"
        assert parse(data, default_category="laptop").products[0].category == "restaurant"

    def test_the_default_fills_a_blank_cell(self):
        data = b"product_name,category,review_text\nAurora,,The screen is dim indeed.\n"
        assert parse(data, default_category="laptop").products[0].category == "laptop"

    def test_rejects_a_category_outside_the_vocabulary(self):
        # The honest limit: the lexicon covers laptops and restaurants only,
        # so a 'phone' catalogue would extract nothing and look broken.
        data = b"product_name,category,review_text\nPixel,phone,The camera is superb here.\n"
        with pytest.raises(CatalogueCsvError, match="not supported"):
            parse(data)

    def test_rejects_an_invalid_default(self):
        data = b"product_name,review_text\nAurora,The screen is dim indeed.\n"
        with pytest.raises(CatalogueCsvError, match="category must be one of"):
            parse(data, default_category="phone")

    def test_rejects_a_file_with_no_category_anywhere(self):
        data = b"product_name,review_text\nAurora,The screen is dim indeed.\n"
        with pytest.raises(CatalogueCsvError, match="not supported"):
            parse(data)


class TestRejections:
    def test_rejects_an_empty_file(self):
        with pytest.raises(CatalogueCsvError, match="empty"):
            parse(b"")

    def test_rejects_a_file_with_no_header(self):
        with pytest.raises(CatalogueCsvError, match="no header row"):
            parse(b"\n")

    def test_rejects_a_header_only_file_by_naming_its_columns(self):
        with pytest.raises(CatalogueCsvError, match="missing required column"):
            parse(b"foo,bar\n")

    def test_names_the_missing_columns_and_what_it_would_accept(self):
        with pytest.raises(CatalogueCsvError) as caught:
            parse(b"foo,bar\n1,2\n")
        message = str(caught.value)
        assert "product_name" in message
        assert "review_text" in message
        # The message has to be actionable, so it lists the accepted headers
        # and what was actually found.
        assert "Accepted headers" in message
        assert "foo" in message

    def test_rejects_a_file_over_the_byte_ceiling(self):
        oversized = b"product_name,category,review_text\n" + b"x" * (MAX_BYTES + 1)
        with pytest.raises(CatalogueCsvError, match="the limit is"):
            parse(oversized)

    def test_rejects_a_file_over_the_row_ceiling(self):
        row = b"Aurora,laptop,The screen is dim indeed and this is long enough.\n"
        data = b"product_name,category,review_text\n" + row * (MAX_ROWS + 1)
        with pytest.raises(CatalogueCsvError, match="more than"):
            parse(data)

    def test_rejects_bytes_that_are_not_utf8(self):
        data = b"product_name,category,review_text\n\xff\xfe,laptop,The screen is dim.\n"
        with pytest.raises(CatalogueCsvError, match="UTF-8"):
            parse(data)

    def test_rejects_a_file_whose_every_row_is_unusable(self):
        data = b"product_name,category,review_text\n,laptop,short\nAurora,laptop,tiny\n"
        with pytest.raises(CatalogueCsvError, match="no usable rows"):
            parse(data)


class TestSkippedRows:
    def test_skips_a_row_with_no_product_name(self):
        data = (
            b"product_name,category,review_text\n"
            b",laptop,The screen is dim indeed.\n"
            b"Aurora,laptop,The keyboard is great here.\n"
        )
        parsed = parse(data)
        assert [p.name for p in parsed.products] == ["Aurora"]
        assert parsed.rows_read == 2
        assert parsed.rows_used == 1

    def test_skips_a_review_too_short_to_carry_an_aspect(self):
        data = (
            b"product_name,category,review_text\n"
            b"Aurora,laptop,bad\n"
            b"Aurora,laptop,The keyboard is great here.\n"
        )
        assert len(parse(data).products[0].reviews) == 1

    def test_drops_a_rating_that_is_not_a_number_or_out_of_range(self):
        data = (
            b"product_name,category,review_text,rating\n"
            b"Aurora,laptop,The screen is dim indeed.,n/a\n"
            b"Aurora,laptop,The keyboard is great here.,87\n"
            b"Aurora,laptop,The battery lasts all day here.,4\n"
        )
        product = parse(data).products[0]
        assert product.rating == 4.0
        assert product.rating_count == 1


class TestCaps:
    def test_keeps_the_most_reviewed_products(self):
        rows = [b"product_name,category,review_text"]
        for index in range(5):
            # Product i gets i+1 reviews, so the ordering is unambiguous.
            for _ in range(index + 1):
                rows.append(b"P%d,laptop,The screen is dim indeed here.n" % index)
        parsed = parse(b"\n".join(rows) + b"\n", max_products=2)
        assert [p.name for p in parsed.products] == ["P4", "P3"]
        assert parsed.truncated_products is True
        assert parsed.products_found == 5

    def test_caps_reviews_per_product_but_still_reports_what_it_saw(self):
        rows = [b"product_name,category,review_text"]
        for index in range(10):
            rows.append(b"Aurora,laptop,The screen is dim indeed number %d.n" % index)
        parsed = parse(b"\n".join(rows) + b"\n", max_reviews=3)
        product = parsed.products[0]
        assert len(product.reviews) == 3
        assert product.reviews_seen == 10
        assert parsed.truncated_reviews is True

    def test_says_nothing_was_truncated_when_it_fitted(self):
        parsed = parse(CANONICAL)
        assert parsed.truncated_products is False
        assert parsed.truncated_reviews is False


class TestSlugs:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Aurora 14", "aurora-14"),
            ("Olive & Ash", "olive-ash"),
            ("Mr. B's Bistro", "mr-b-s-bistro"),
            ("  spaced  out  ", "spaced-out"),
            ("!!!", "product"),
            ("", "product"),
        ],
    )
    def test_slugify(self, name, expected):
        assert slugify(name) == expected

    def test_two_products_with_the_same_name_get_distinct_slugs(self):
        # Two restaurants really can share a name, and a duplicate slug would
        # make one shadow the other on its own product page.
        data = (
            b"product_name,category,review_text\n"
            b"Joe's,restaurant,Great food but the service was slow.\n"
            b"Joe's ,restaurant,The atmosphere here is wonderful truly.\n"
        )
        parsed = parse(data)
        slugs = [p.slug for p in parsed.products]
        assert len(slugs) == len(set(slugs))
