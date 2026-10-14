"""Redaction of key-shaped strings in model output, and of secrets in logs."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[redacted]"

# Vendor API keys: Anthropic (`sk-ant-api03-…`), OpenAI project keys (`sk-proj-…`) and any
# other `sk-` key, whose bodies hold `-` and `_`. Not when glued to a word or a hyphen
# (`task-…`, `nordic-sk-race-…`). A key after `=`, `:`, a quote, a space or a line break is.
_KEY_START = r"(?<![\w-])"
API_KEY_PATTERNS = (
    _KEY_START + r"sk-ant-[A-Za-z0-9_-]{20,}",
    _KEY_START + r"sk-proj-[A-Za-z0-9_-]{20,}",
    _KEY_START + r"sk-[A-Za-z0-9_-]{20,}",
)
DEFAULT_PATTERNS = (
    *API_KEY_PATTERNS,
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class Redactor:
    """Replaces key-shaped strings and non-allowlisted emails with `[redacted]`."""

    def __init__(self, patterns: Iterable[str] = (), email_allowlist: Iterable[str] = ()) -> None:
        self._patterns = [re.compile(p) for p in (*DEFAULT_PATTERNS, *patterns)]
        self._allowed = {e.lower() for e in email_allowlist}

    def redact(self, text: str) -> str:
        for pattern in self._patterns:
            text = pattern.sub(REDACTED, text)
        return EMAIL.sub(
            lambda m: m.group(0) if m.group(0).lower() in self._allowed else REDACTED, text
        )


# `"secret": "..."`, `secret=...` and `Bearer ag_xxx:...` shapes, whatever the value.
_SECRET_SHAPES = (
    re.compile(r'("secret"\s*:\s*")[^"]*(")'),
    re.compile(r"(\bsecret=)[^\s&,;]+()"),
    re.compile(r"(Bearer\s+ag_[a-z2-7]{6}:)[^\s\"']+()"),
    *(re.compile(f"(){pattern}()") for pattern in API_KEY_PATTERNS),
)


class SecretFilter(logging.Filter):
    """Scrubs known secret values and secret-shaped substrings from log records."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def add_secret(self, secret: str) -> None:
        if secret:
            self._secrets.add(secret)

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        for shape in _SECRET_SHAPES:
            text = shape.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(2)}", text)
        return text

    def scrub_values(self, value: Any) -> Any:
        """`scrub` applied to every string inside a JSON-shaped value, before it is dumped:
        once dumped, a key after a line break follows the escape's `n`, a word character."""
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return {key: self.scrub_values(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self.scrub_values(item) for item in value]
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        scrubbed = self.scrub(message)
        if scrubbed != message:
            record.msg, record.args = scrubbed, None
        return True


# One filter for the whole package; the CLI adds the loaded secret to it, and
# `config.api_key()` the API key of API-key mode.
secret_filter = SecretFilter()
