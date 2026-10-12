"""
app/services/ncert_subjects.py — one subject filter for every NCERT table query.

Subjects are matched EXACTLY (case-insensitive), never as a substring: "%Science%" also matched
"Political Science", so Class 9/10 Science dropdowns and retrieval pulled civics chapters.
A few subjects are stored under several names (e.g. "Physics", "Physics-I", "Physics-II");
those map to their whole family.
"""

from typing import List, Optional

_FAMILIES = {
    "science": ["Science"],
    "physics": ["Physics", "Physics-I", "Physics-II"],
    "physics-i": ["Physics", "Physics-I", "Physics-II"],
    "physics-ii": ["Physics", "Physics-I", "Physics-II"],
    "chemistry": ["Chemistry", "Chemistry-I", "Chemistry-II"],
    "chemistry-i": ["Chemistry", "Chemistry-I", "Chemistry-II"],
    "chemistry-ii": ["Chemistry", "Chemistry-I", "Chemistry-II"],
    "political science": ["Political Science", "Political_Science"],
    "political_science": ["Political Science", "Political_Science"],
    "pol science": ["Political Science", "Political_Science"],
}


def ncert_subject_names(subject: str) -> Optional[List[str]]:
    """Exact DB subject names for a subject family, or None for a single case-insensitive match."""
    return _FAMILIES.get((subject or "").strip().lower())


def apply_ncert_subject_filter(query, subject: str):
    names = ncert_subject_names(subject)
    if names:
        return query.in_("subject", names)
    # ILIKE without wildcards = case-insensitive equality; escape LIKE metacharacters
    clean = (subject or "").strip().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return query.ilike("subject", clean)
