import re


def remove_emails(text: str) -> tuple[str, int]:
    """
    Remove email addresses from text using regex.

    Matches common email patterns like:
    - user@domain.com
    - first.last@subdomain.domain.co.uk
    - user+tag@domain.org

    Returns:
        tuple[str, int]: (masked text, count of emails removed)
    """
    # Email regex pattern
    # Matches: word characters, dots, hyphens, plus signs @ domain with subdomains
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    # Count matches before replacing
    count = len(re.findall(email_pattern, text))

    # Replace emails with [EMAIL]
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)

    return masked_text, count


def remove_phone_numbers(text: str) -> tuple[str, int]:
    """
    Remove phone numbers from text using regex.

    Matches various phone number formats like:
    - (123) 456-7890
    - 123-456-7890
    - 123.456.7890
    - +1 123 456 7890
    - 1234567890

    Returns:
        tuple[str, int]: (masked text, count of phone numbers removed)
    """
    # List of phone number patterns to match different formats
    phone_patterns = [
        # International format: +1 234 567 8900, +12345678900
        r"\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,4}",
        r"\+\d{10,15}",
        # US format with parentheses: (123) 456-7890
        r"\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}",
        # Standard formats: 123-456-7890, 123.456.7890, 123 456 7890
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        # 10-digit number: 1234567890
        r"\b\d{10}\b",
        # Extension formats: 123-456-7890 x1234
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\s?(?:ext|x|extension)\s?\d{1,5}\b",
    ]

    # Combine all patterns with OR operator
    combined_pattern = "|".join(phone_patterns)

    # Count matches before replacing
    count = len(re.findall(combined_pattern, text, flags=re.IGNORECASE))

    # Replace phone numbers with [PHONE]
    masked_text = re.sub(combined_pattern, "|||PHONE_NUMBER|||", text, flags=re.IGNORECASE)

    return masked_text, count


def remove_ip_addresses(text: str) -> tuple[str, int]:
    """
    Remove IP addresses from text using regex.

    Matches:
    - IPv4 addresses: 192.168.1.1
    - IPv6 addresses: 2001:0db8:85a3:0000:0000:8a2e:0370:7334

    Returns:
        tuple[str, int]: (masked text, count of IP addresses removed)
    """
    # IPv4 pattern: 4 groups of 1-3 digits separated by dots
    ipv4_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    # IPv6 pattern (simplified - matches most common formats)
    # Full IPv6: 8 groups of 4 hex digits separated by colons
    # Compressed IPv6: allows :: for consecutive zeros
    ipv6_pattern = r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b|\b(?:[A-Fa-f0-9]{1,4}:)*:(?:[A-Fa-f0-9]{1,4}:)*[A-Fa-f0-9]{1,4}\b"

    # Count matches before replacing
    ipv6_count = len(re.findall(ipv6_pattern, text))
    ipv4_count = len(re.findall(ipv4_pattern, text))
    total_count = ipv6_count + ipv4_count

    # First remove IPv6 (more specific pattern)
    text = re.sub(ipv6_pattern, "|||IP_ADDRESS|||", text)
    # Then remove IPv4
    text = re.sub(ipv4_pattern, "|||IP_ADDRESS|||", text)

    return text, total_count
