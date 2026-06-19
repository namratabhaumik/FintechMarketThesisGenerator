"""Text utilities for content cleaning and normalization."""

import re

# Promotional/ad phrases to filter out
_AD_PATTERNS = re.compile(
    r'register now|early bird|save up to|\$\d+ off|buy tickets|get tickets|'
    r'sign up|subscribe now|learn more|click here|limited time',
    re.IGNORECASE
)

# Boilerplate patterns: contact info, event promotions, author bios, newsletter prompts
_BOILERPLATE_PATTERNS = re.compile(
    r'you can contact.*?(?:\n|$)|'  # Contact blocks
    r'email:.*?(?:\n|$)|'  # Email addresses
    r'discover your next.*?(?:\n|$)|'  # Event promotions
    r'hear from \d+\+.*?(?:\n|$)|'  # Event speaker counts
    r'by \w+ \w+\s*(?:\n|$)|'  # Author bylines (e.g., "By John Doe")
    r'about the author.*?(?:\n\n|$)|'  # Author bio headers
    r'subscribe.*?newsletter.*?(?:\n|$)|'  # Newsletter subscriptions
    r'follow us on.*?(?:\n|$)|'  # Social media follows
    r'visit.*?website.*?(?:\n|$)',  # Website visit prompts
    re.IGNORECASE | re.DOTALL
)


def clean_article_text(text: str) -> str:
    """Remove noise from article text (ads, boilerplate, excess whitespace).

    Strips marketing language, structural boilerplate (contact info, author bios,
    event promotions), and normalizes whitespace before the text is embedded or
    summarized. This upstream cleaning improves signal-to-noise ratio for all
    downstream processing.

    Args:
        text: Raw article text potentially containing noise.

    Returns:
        Cleaned text with ad/promo phrases, boilerplate, and excess whitespace removed.
    """
    # Nothing to clean (empty or None-like) -> hand it straight back.
    if not text:
        return text

    # The cleaning runs as a pipeline, each step rewriting `text` in turn:
    # 1) drop ad/promo phrases, replacing each with a space so neighbouring
    #    words don't fuse together ("...endbuy ticketsstart..." -> "...end start...").
    text = _AD_PATTERNS.sub(' ', text)

    # 2) drop structural boilerplate (contact blocks, event promos, author bios).
    text = _BOILERPLATE_PATTERNS.sub(' ', text)

    # 3) tidy the whitespace the removals left behind: runs of spaces collapse to
    #    one, runs of blank lines collapse to one newline, and each line is then
    #    stripped of leading/trailing spaces.
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n\s*\n+', '\n', text)  # Multiple blank lines → single newline
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()
