"""
app/core/sanitize.py — Input sanitization utilities

Use these in ALL endpoints that take user input for DB queries.
"""

import re
import html


def sanitize_like(value: str, max_length: int = 200) -> str:
    """
    Sanitize a string before using in Supabase .ilike() queries.
    Removes SQL LIKE wildcards and limits length.
    """
    if not value:
        return ""
    value = value.strip()
    # Remove LIKE special characters
    value = re.sub(r'[%_\\]', '', value)
    # Limit length
    return value[:max_length]


def sanitize_text(value: str, max_length: int = 5000) -> str:
    """
    Sanitize free-text input (exam titles, question text, etc.).
    Escapes HTML to prevent stored XSS.
    """
    if not value:
        return ""
    value = value.strip()
    value = html.escape(value)
    return value[:max_length]


def sanitize_uuid(value: str) -> str:
    """
    Validate UUID format. Raises ValueError if invalid.
    Prevents injection via user_id / test_id fields.
    """
    if not value:
        raise ValueError("ID cannot be empty")
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE,
    )
    if not uuid_pattern.match(value):
        raise ValueError(f"Invalid ID format")
    return value.lower()


def validate_class_grade(value: str) -> str:
    """Only allow valid class grades: 1-12."""
    match = re.search(r'\d+', value)
    if not match:
        raise ValueError("Invalid class grade")
    grade = int(match.group())
    if grade < 1 or grade > 12:
        raise ValueError("Class grade must be between 1 and 12")
    return str(grade)