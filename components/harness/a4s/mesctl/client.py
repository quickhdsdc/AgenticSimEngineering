"""HTTP boundary for OpenMES. Credentials stay in the configured environment."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx


class MesError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, details: Any = None):
        super().__init__(message)
        self.status = status
        self.details = details


@dataclass
class Config:
    endpoint: str
    token: str = field(default="", repr=False)
    timeout: float = 15.0
    verify_tls: bool = True
    server_timezone: str = "UTC"

    def __post_init__(self):
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise MesError(
                "A4S_MES_ENDPOINT must be an HTTP(S) base URL without credentials, query or fragment"
            )
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise MesError("A4S_MES_TIMEOUT must be a positive finite number")
        self.endpoint = self.endpoint.rstrip("/")

    @classmethod
    def from_env(cls):
        endpoint = os.getenv("A4S_MES_ENDPOINT", "").strip()
        if not endpoint:
            raise MesError("Set A4S_MES_ENDPOINT to the OpenMES base URL (without /api/v1)")
        try:
            timeout = float(os.getenv("A4S_MES_TIMEOUT", "15"))
        except ValueError:
            raise MesError("A4S_MES_TIMEOUT must be a number") from None
        return cls(
            endpoint,
            os.getenv("A4S_MES_TOKEN", ""),
            timeout,
            os.getenv("A4S_MES_VERIFY_TLS", "true").lower() not in ("0", "false", "no"),
            os.getenv("A4S_MES_TIMEZONE", ""),
        )


class MesClient:
    def __init__(self, config: Config, *, transport=None):
        self.config = config
        self.execution_context = None
        self.http = httpx.Client(
            base_url=config.endpoint + "/",
            verify=config.verify_tls,
            timeout=httpx.Timeout(config.timeout, connect=min(5, config.timeout)),
            follow_redirects=False,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.http.close()

    def clean(self, value: Any) -> Any:
        """Redact credential fields and an echoed token, including error responses."""
        if isinstance(value, dict):
            return {
                key: "[REDACTED]"
                if any(
                    word in key.lower()
                    for word in ("password", "token", "secret", "authorization", "api_key")
                )
                else self.clean(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.clean(item) for item in value]
        if isinstance(value, str) and self.config.token:
            return value.replace(self.config.token, "[REDACTED]")
        return value

    def request(self, method: str, path: str, *, params=None, body=None):
        if not path.startswith("/api/") or ".." in path or "?" in path or "#" in path:
            raise MesError("Invalid OpenMES API path")
        headers = self.execution_context.headers() if self.execution_context else {}
        if path != "/api/health":
            if not self.config.token:
                raise MesError("Set A4S_MES_TOKEN to a dedicated OpenMES user's Sanctum token")
            headers["Authorization"] = f"Bearer {self.config.token}"
        try:
            response = self.http.request(
                method, path.lstrip("/"), params=params, json=body, headers=headers
            )
        except httpx.HTTPError:
            hint = (
                " Mutation outcome is unknown; read the resource before retrying."
                if method != "GET"
                else ""
            )
            raise MesError("OpenMES connection failed or timed out." + hint) from None
        if response.status_code == 204:
            return None
        try:
            data = self.clean(response.json())
        except ValueError:
            raise MesError(
                "OpenMES returned a non-JSON response; check the base URL and proxy",
                status=response.status_code,
            ) from None
        if not 200 <= response.status_code < 300:
            message = (
                data.get("message", "OpenMES request failed")
                if isinstance(data, dict)
                else "OpenMES request failed"
            )
            raise MesError(str(message), status=response.status_code, details=data)
        return data

    def get(self, path, **params):
        return self.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def rows(self, path, **params):
        """Read list envelopes, including Laravel's paginator nested under data."""
        rows = []
        for page in range(1, 1001):
            result = self.get(path, **{**params, "page": page})
            data = result.get("data") if isinstance(result, dict) else None
            meta = result.get("meta", {}) if isinstance(result, dict) else {}
            if isinstance(data, dict):
                meta, data = data, data.get("data")
            if not isinstance(data, list):
                raise MesError("Unexpected OpenMES list response")
            rows.extend(data)
            if page >= int(meta.get("last_page", 1)):
                return rows
        raise MesError("OpenMES pagination exceeded 1000 pages; narrow the query")

    def resolve(self, collection, ref, field):
        if str(ref).isdigit() and int(ref) > 0:
            return int(ref)
        matches = [row for row in self.rows(f"/api/v1/{collection}") if row.get(field) == ref]
        if len(matches) != 1:
            raise MesError(
                f"Expected exactly one {collection} match for {ref!r}; found {len(matches)}. Use a numeric ID."
            )
        return int(matches[0]["id"])
