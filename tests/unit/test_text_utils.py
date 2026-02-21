"""Unit tests for text_utils.clean_article_text."""

import pytest
from core.utils.text_utils import clean_article_text


class TestCleanArticleText:
    """Tests for the clean_article_text utility function."""

    # === Empty / None Input ===

    def test_empty_string_returns_empty(self):
        """Empty input returns empty string unchanged."""
        assert clean_article_text("") == ""

    def test_none_input_returns_none(self):
        """None input is returned as-is (falsy early return)."""
        assert clean_article_text(None) is None

    # === Ad / Promo Phrase Removal ===

    def test_removes_sign_up(self):
        result = clean_article_text("The product is great. Sign up today for access.")
        assert "sign up" not in result.lower()

    def test_removes_subscribe_now(self):
        result = clean_article_text("Subscribe now to get the latest fintech news.")
        assert "subscribe now" not in result.lower()

    def test_removes_click_here(self):
        result = clean_article_text("Click here to read more about the acquisition.")
        assert "click here" not in result.lower()

    def test_removes_learn_more(self):
        result = clean_article_text("Learn more about our enterprise payment solutions.")
        assert "learn more" not in result.lower()

    def test_removes_limited_time(self):
        result = clean_article_text("Limited time offer: get 50% off the annual plan.")
        assert "limited time" not in result.lower()

    def test_removes_register_now(self):
        result = clean_article_text("Register now for the fintech summit.")
        assert "register now" not in result.lower()

    def test_removes_early_bird(self):
        result = clean_article_text("Early bird tickets available for the conference.")
        assert "early bird" not in result.lower()

    def test_ad_removal_is_case_insensitive(self):
        result = clean_article_text("SIGN UP now and CLICK HERE for details.")
        assert "sign up" not in result.lower()
        assert "click here" not in result.lower()

    # === Boilerplate Removal ===

    def test_removes_about_the_author(self):
        article = "Stripe raised $1B.\n\nAbout the author\nJohn covers fintech at TechCrunch.\n\n"
        result = clean_article_text(article)
        assert "about the author" not in result.lower()

    def test_removes_follow_us_on(self):
        result = clean_article_text("The deal closed yesterday.\nFollow us on Twitter for updates.\n")
        assert "follow us on" not in result.lower()

    def test_removes_email_lines(self):
        result = clean_article_text("Revenue grew 40%.\nemail: contact@example.com\n")
        assert "email:" not in result.lower()

    # === Whitespace Normalisation ===

    def test_collapses_multiple_spaces(self):
        result = clean_article_text("Fintech    is    growing   fast.")
        assert "  " not in result  # no double spaces

    def test_collapses_multiple_blank_lines(self):
        result = clean_article_text("First paragraph.\n\n\n\nSecond paragraph.")
        assert "\n\n" not in result  # no consecutive blank lines

    def test_strips_leading_trailing_whitespace(self):
        result = clean_article_text("  \n  Fintech news.  \n  ")
        assert result == result.strip()

    def test_strips_per_line_whitespace(self):
        result = clean_article_text("  line one  \n  line two  ")
        for line in result.split("\n"):
            assert line == line.strip()

    # === Preservation of Meaningful Content ===

    def test_preserves_core_article_text(self):
        article = "Stripe raised $1 billion in Series H funding. The company plans to expand into Southeast Asia."
        result = clean_article_text(article)
        assert "Stripe" in result
        assert "billion" in result
        assert "Southeast Asia" in result

    def test_does_not_destroy_normal_sentences(self):
        article = "Digital payments are growing rapidly across emerging markets."
        result = clean_article_text(article)
        assert "Digital payments" in result
        assert "emerging markets" in result
