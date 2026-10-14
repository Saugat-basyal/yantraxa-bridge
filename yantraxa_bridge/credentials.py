"""`$YANTRAXA_HOME/credentials` (`{ agent_id, secret }`, 0600) and the bridge's key pair."""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from .config import ConfigError, ensure_home, write_private, yantraxa_home
from .protocol import AGENT_ID_PATTERN


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(pattern=AGENT_ID_PATTERN)
    secret: SecretStr

    def secret_value(self) -> str:
        return self.secret.get_secret_value()

    def bearer(self) -> str:
        return f"Bearer {self.agent_id}:{self.secret_value()}"


def credentials_path(home: Path | None = None) -> Path:
    return (home or yantraxa_home()) / "credentials"


def _tighten(path: Path) -> None:
    if stat.S_IMODE(os.stat(path).st_mode) != 0o600:
        os.chmod(path, 0o600)


def load_credentials(home: Path | None = None) -> Credentials | None:
    path = credentials_path(home)
    if not path.exists():
        return None
    _tighten(path)
    try:
        return Credentials.model_validate(json.loads(path.read_text()))
    except (ValueError, ValidationError) as error:
        raise ConfigError(f"{path} is unreadable; pair again") from error


def save_credentials(credentials: Credentials, home: Path | None = None) -> Path:
    home = ensure_home(home)
    path = credentials_path(home)
    body = {"agent_id": credentials.agent_id, "secret": credentials.secret_value()}
    write_private(path, json.dumps(body) + "\n")
    return path


def delete_credentials(home: Path | None = None) -> bool:
    path = credentials_path(home)
    if not path.exists():
        return False
    path.unlink()
    return True


def key_path(home: Path | None = None) -> Path:
    return (home or yantraxa_home()) / "bridge_key.pem"


def load_or_create_public_key(home: Path | None = None) -> str:
    """The bridge's Ed25519 public key, base64 of the raw 32 bytes.

    The private key stays in `bridge_key.pem` (0600). The server shows its fingerprint
    so the owner can tell which bridge they are approving.
    """
    path = key_path(home)
    if path.exists():
        _tighten(path)
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ConfigError(f"{path} is not an Ed25519 key")
    else:
        ensure_home(home)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        write_private(path, pem.decode())
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()
