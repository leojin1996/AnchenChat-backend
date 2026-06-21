from __future__ import annotations


class AuthError(Exception):
    """Domain error for auth subsystem; carries a stable code and human message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AuthConfigError(Exception):
    """Raised when auth subsystem cannot be initialized (e.g. missing secret)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
