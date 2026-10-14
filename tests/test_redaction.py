import logging

import pytest

from yantraxa_bridge.credentials import Credentials
from yantraxa_bridge.redaction import REDACTED, Redactor, SecretFilter


def test_default_patterns_redact_key_shaped_strings():
    redactor = Redactor()
    text = (
        "openai sk-abcdefghijklmnopqrstuvwxyz123 aws AKIAABCDEFGHIJKLMNOP "
        "-----BEGIN RSA PRIVATE KEY----- mail me at bob@example.com"
    )
    out = redactor.redact(text)
    assert "sk-abc" not in out
    assert "AKIA" not in out
    assert "PRIVATE KEY" not in out
    assert "bob@example.com" not in out
    assert out.count(REDACTED) == 4


def test_short_sk_prefix_is_left_alone():
    assert Redactor().redact("ask-me later") == "ask-me later"


def test_allowlisted_emails_survive():
    redactor = Redactor(email_allowlist=["team@yantraxa.test"])
    out = redactor.redact("write to team@yantraxa.test or eve@evil.test")
    assert "team@yantraxa.test" in out
    assert "eve@evil.test" not in out


def test_extra_patterns_are_added():
    redactor = Redactor(patterns=[r"project-[0-9]{4}"])
    assert redactor.redact("see project-1234 and sk-abcdefghijklmnopqrstuvwxyz") == (
        f"see {REDACTED} and {REDACTED}"
    )


def test_secret_filter_scrubs_log_records(caplog):
    logger = logging.getLogger("yantraxa_bridge.test")
    filt = SecretFilter()
    filt.add_secret("hunter2hunter2")
    logger.addFilter(filt)
    try:
        with caplog.at_level(logging.INFO, logger="yantraxa_bridge.test"):
            logger.info("auth with %s", "hunter2hunter2")
            logger.info('frame {"secret": "other-value", "x": 1}')
            logger.info("secret=abc123 next")
    finally:
        logger.removeFilter(filt)
    text = caplog.text
    assert "hunter2hunter2" not in text
    assert "other-value" not in text
    assert "abc123" not in text
    assert text.count(REDACTED) == 3


def test_credentials_never_show_the_secret():
    creds = Credentials(agent_id="ag_abc234", secret="topsecretvalue")
    assert "topsecretvalue" not in repr(creds)
    assert "topsecretvalue" not in str(creds)
    assert creds.secret_value() == "topsecretvalue"


ANTHROPIC_KEY = "sk-ant-api03-" + "Ab3_dE-fG" * 10 + "-AA"
OPENAI_PROJECT_KEY = "sk-proj-" + "x1Y_2z-" * 8
OPENAI_KEY = "sk-" + "a1B2_c3D-" * 4


@pytest.mark.parametrize("key", [ANTHROPIC_KEY, OPENAI_PROJECT_KEY, OPENAI_KEY])
def test_vendor_api_keys_with_dashes_are_redacted_whole(key):
    out = Redactor().redact(f"my key is {key}, keep it safe")
    assert out == f"my key is {REDACTED}, keep it safe"
    assert SecretFilter().scrub(f"export KEY={key}") == f"export KEY={REDACTED}"


def test_words_that_merely_contain_sk_dash_are_left_alone():
    text = (
        "the task-management-and-planning-session-notes and ask-me-anything-about-the-plan "
        "and nordic-sk-race-condition-notes-for-the-team"
    )
    assert Redactor().redact(text) == text
    assert SecretFilter().scrub(text) == text


@pytest.mark.parametrize("before", ["", "KEY=", "key:", '"', "'", " ", "\n", "\t", "(", "[", ","])
def test_a_key_after_punctuation_space_or_a_line_break_is_redacted(before):
    assert Redactor().redact(f"{before}{OPENAI_KEY}") == f"{before}{REDACTED}"
    assert SecretFilter().scrub(f"{before}{ANTHROPIC_KEY}") == f"{before}{REDACTED}"


def test_the_api_key_of_api_key_mode_reaches_the_log_filter(monkeypatch, caplog):
    from yantraxa_bridge.config import BridgeConfig, api_key
    from yantraxa_bridge.redaction import secret_filter

    odd = "a-key-with-no-known-shape-123"  # only the registered value can catch this
    monkeypatch.setenv("MY_VENDOR_KEY", odd)
    assert api_key(BridgeConfig(adapter="claude-code", api_key_env="MY_VENDOR_KEY")) == odd
    assert secret_filter.scrub(f"starting with {odd}") == f"starting with {REDACTED}"
    logger = logging.getLogger("yantraxa_bridge.test.key")
    logger.addFilter(secret_filter)
    try:
        with caplog.at_level(logging.INFO, logger="yantraxa_bridge.test.key"):
            logger.info("env had %s", odd)
    finally:
        logger.removeFilter(secret_filter)
    assert odd not in caplog.text
