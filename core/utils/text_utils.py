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
    r'\w+ (?:covers|writes about|is a|reports on).*?(?:\n|$)|' 
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
    if not text:
        return text

    # Remove ad/promo phrases inline (replace with space to avoid word concatenation)
    text = _AD_PATTERNS.sub(' ', text)

    # Remove boilerplate content (contact blocks, event promos, author bios, etc.)
    text = _BOILERPLATE_PATTERNS.sub(' ', text)

    # Normalize whitespace: collapse multiple spaces, strip per line
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n\s*\n+', '\n', text)  # Multiple blank lines → single newline
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()
