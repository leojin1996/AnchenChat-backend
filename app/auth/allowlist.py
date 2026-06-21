from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.auth.errors import AuthConfigError

_CN_PHONE_RE = re.compile(r"^1[3-9]\d{9}$")


@dataclass(frozen=True)
class AllowlistEntry:
    phone: str
    name: str
    role: str = "user"


def normalize_phone(value: str) -> str:
    """Strip whitespace and a leading +86/86 prefix; raise when not a CN mobile."""

    if value is None:
        raise ValueError("phone is required")
    candidate = str(value).strip().replace(" ", "").replace("-", "")
    if candidate.startswith("+86"):
        candidate = candidate[3:]
    elif candidate.startswith("86") and len(candidate) == 13:
        candidate = candidate[2:]
    if not _CN_PHONE_RE.match(candidate):
        raise ValueError("invalid Chinese mobile phone number")
    return candidate


class Allowlist:
    """Immutable view of phone-number whitelist loaded from YAML."""

    def __init__(self, entries: list[AllowlistEntry]) -> None:
        self._by_phone: dict[str, AllowlistEntry] = {entry.phone: entry for entry in entries}

    def __len__(self) -> int:
        return len(self._by_phone)

    def get(self, phone: str) -> AllowlistEntry | None:
        try:
            normalized = normalize_phone(phone)
        except ValueError:
            return None
        return self._by_phone.get(normalized)

    def is_allowed(self, phone: str) -> bool:
        return self.get(phone) is not None

    def entries(self) -> list[AllowlistEntry]:
        return list(self._by_phone.values())


def load_allowlist(path: str | Path) -> Allowlist:
    """Read an allowlist YAML file. Missing/malformed input raises AuthConfigError."""

    file_path = Path(path)
    if not file_path.exists():
        raise AuthConfigError(f"allowlist file not found: {file_path}")
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise AuthConfigError(f"failed to parse allowlist YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise AuthConfigError("allowlist root must be a mapping")
    raw_users = raw.get("users", [])
    if not isinstance(raw_users, list):
        raise AuthConfigError("allowlist 'users' must be a list")

    entries: list[AllowlistEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_users):
        if not isinstance(item, dict):
            raise AuthConfigError(f"allowlist user #{index} must be a mapping")
        raw_phone = item.get("phone")
        if raw_phone is None:
            raise AuthConfigError(f"allowlist user #{index} missing 'phone'")
        try:
            phone = normalize_phone(raw_phone)
        except ValueError as exc:
            raise AuthConfigError(
                f"allowlist user #{index} has invalid phone {raw_phone!r}: {exc}"
            ) from exc
        if phone in seen:
            raise AuthConfigError(f"allowlist contains duplicate phone {phone}")
        seen.add(phone)
        name = str(item.get("name") or phone)
        role = str(item.get("role") or "user")
        entries.append(AllowlistEntry(phone=phone, name=name, role=role))

    return Allowlist(entries)
