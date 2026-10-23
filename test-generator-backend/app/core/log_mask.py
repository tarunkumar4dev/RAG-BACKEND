"""
Student data never reaches the logs in clear: names, emails and phone numbers are masked.

  mask_name("Asha Verma")         → "A*** V***"
  mask_email("asha.v@gmail.com")  → "a***@gmail.com"
  mask_phone("+91 98765 43210")   → "********3210"
  PIIMaskFilter                   → a logging filter that masks any email / phone number left in a message
"""

import logging
import re

_EMAIL = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
# Indian mobile numbers with or without +91 / 0, spaces or dashes inside (10 digits starting 6-9)
_PHONE = re.compile(r"(?<![\w-])(?:\+?91[\s-]?|0)?([6-9](?:[\s-]?\d){9})(?![\w-])")


def mask_name(name) -> str:
    parts = str(name or "").split()
    return " ".join(p[0] + "***" for p in parts) if parts else ""


def mask_email(email) -> str:
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", str(email or ""))


def mask_phone(phone) -> str:
    digits = re.sub(r"\D", "", str(phone or ""))
    return "*" * max(len(digits) - 4, 0) + digits[-4:] if digits else ""


def mask_text(text: str) -> str:
    text = _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)
    return _PHONE.sub(lambda m: mask_phone(m.group(0)), text)


class PIIMaskFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        masked = mask_text(msg)
        if masked != msg:
            record.msg, record.args = masked, None
        return True
