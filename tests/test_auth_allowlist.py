from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.auth.allowlist import Allowlist, AllowlistEntry, load_allowlist, normalize_phone
from app.auth.errors import AuthConfigError


def test_normalize_phone_accepts_plain_number() -> None:
    assert normalize_phone(" 138 0013 8000 ") == "13800138000"


def test_normalize_phone_strips_country_code() -> None:
    assert normalize_phone("+8613800138000") == "13800138000"
    assert normalize_phone("8613900139000") == "13900139000"


def test_normalize_phone_rejects_invalid_inputs() -> None:
    for value in ["", "1234567890", "1234567890a", "12000000000"]:
        with pytest.raises(ValueError):
            normalize_phone(value)


def test_allowlist_lookup_and_membership() -> None:
    allowlist = Allowlist(
        [
            AllowlistEntry(phone="13800138000", name="店长", role="admin"),
            AllowlistEntry(phone="13900139000", name="店员"),
        ]
    )
    assert allowlist.is_allowed("+8613800138000")
    entry = allowlist.get("13900139000")
    assert entry is not None
    assert entry.role == "user"
    assert allowlist.get("13700137000") is None


def test_load_allowlist_parses_yaml(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.yaml"
    path.write_text(
        textwrap.dedent(
            """
            users:
              - phone: "+8613800138000"
                name: "店长"
                role: "admin"
              - phone: 13900139000
            """
        ).strip(),
        encoding="utf-8",
    )

    allowlist = load_allowlist(path)
    assert len(allowlist) == 2
    entries = allowlist.entries()
    assert entries[0].phone == "13800138000"
    assert entries[0].role == "admin"
    assert entries[1].name == "13900139000"
    assert entries[1].role == "user"


def test_load_allowlist_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AuthConfigError):
        load_allowlist(tmp_path / "missing.yaml")


def test_load_allowlist_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text(
        "users:\n  - phone: '13800138000'\n  - phone: '13800138000'\n",
        encoding="utf-8",
    )
    with pytest.raises(AuthConfigError):
        load_allowlist(path)


def test_load_allowlist_rejects_invalid_phone(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("users:\n  - phone: '12345'\n", encoding="utf-8")
    with pytest.raises(AuthConfigError):
        load_allowlist(path)


def test_load_allowlist_rejects_malformed_root(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 13800138000\n", encoding="utf-8")
    with pytest.raises(AuthConfigError):
        load_allowlist(path)
