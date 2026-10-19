"""Synchronous BaSyx V3 REST client.

Synchronous on purpose. The CLI is a short-lived process invoked from a
shell, so an event loop buys nothing and costs a layer of traceback. The
harness gets its concurrency from running several CLI processes, not from
async inside one.

Endpoints follow AAS Part 2 (the "AAS Repository" / "Submodel Repository"
profiles that BaSyx implements).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .paths import ElementPath, encode_id

DEFAULT_TIMEOUT = 30.0

# Connecting and reading are different waits and deserve different patience.
# A repository that will not accept a TCP connection is down, and no amount of
# waiting changes that -- while a repository that *has* answered may legitimately
# take twenty seconds over a large submodel. Windows' own connect timeout is 21s,
# which is what a mistyped endpoint in the web UI cost before this: twenty-one
# seconds of a frozen page to learn that a host does not exist.
CONNECT_TIMEOUT = 5.0


class AasHttpError(RuntimeError):
    """A BaSyx call that came back non-2xx.

    Carries the server's message verbatim. The agent reads these strings, so
    they must stay specific: "404 on Nameplate.Foo" tells it to check the
    path, "409" tells it the element already exists, and the two lead to
    different next commands.
    """

    def __init__(self, status: int, url: str, body: str) -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} on {url}: {body[:500]}")


@dataclass
class Endpoint:
    """Where an AAS lives.

    A4S deployments routinely address two environments at once: the
    agent-internal one holding the SMT collection, and the customer
    environment supplied per task. Keeping this a value rather than a global
    is what stops the two being conflated.
    """

    base_url: str
    token: Optional[str] = None
    verify_tls: bool = False

    @classmethod
    def from_env(cls, override: Optional[str] = None) -> "Endpoint":
        base = override or os.getenv("A4S_AAS_ENDPOINT")
        if not base:
            raise RuntimeError(
                "No AAS endpoint. Pass --endpoint or set A4S_AAS_ENDPOINT "
                "(deployments differ per host, so there is no default)."
            )
        verify = os.getenv("A4S_AAS_SSL_VERIFY", "false").strip().lower() in (
            "1", "true", "yes", "on",
        )
        return cls(base_url=base.rstrip("/"), token=os.getenv("A4S_AAS_TOKEN"), verify_tls=verify)


class BasyxClient:
    def __init__(self, endpoint: Endpoint, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.endpoint = endpoint
        headers = {"Accept": "application/json"}
        if endpoint.token:
            headers["Authorization"] = f"Bearer {endpoint.token}"
        self._http = httpx.Client(
            base_url=endpoint.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT, timeout)),
            verify=endpoint.verify_tls,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "BasyxClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport -------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise AasHttpError(response.status_code, str(response.request.url), response.text)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: Any) -> Any:
        return self._request("POST", path, json=payload)

    def put(self, path: str, payload: Any) -> Any:
        return self._request("PUT", path, json=payload)

    def patch(self, path: str, payload: Any) -> Any:
        return self._request("PATCH", path, json=payload)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # -- shells ----------------------------------------------------------

    def list_shells(self, limit: int = 100) -> List[Dict[str, Any]]:
        body = self.get("/shells", params={"limit": limit})
        return _paged(body)

    def get_shell(self, shell_id: str) -> Dict[str, Any]:
        return self.get(f"/shells/{encode_id(shell_id)}")

    def create_shell(self, shell: Dict[str, Any]) -> Dict[str, Any]:
        return self.post("/shells", shell)

    def delete_shell(self, shell_id: str) -> None:
        self.delete(f"/shells/{encode_id(shell_id)}")

    def list_submodel_refs(self, shell_id: str) -> List[Dict[str, Any]]:
        body = self.get(f"/shells/{encode_id(shell_id)}/submodel-refs")
        return _paged(body)

    def add_submodel_ref(self, shell_id: str, reference: Dict[str, Any]) -> Any:
        return self.post(f"/shells/{encode_id(shell_id)}/submodel-refs", reference)

    def remove_submodel_ref(self, shell_id: str, submodel_id: str) -> None:
        """Unbind a submodel from a shell, leaving the submodel itself alone.

        Both identifiers are base64url-encoded: AAS Part 2 encodes every
        Identifier appearing in a path segment, and this endpoint has two.
        """
        self.delete(
            f"/shells/{encode_id(shell_id)}/submodel-refs/{encode_id(submodel_id)}"
        )

    def replace_submodel(self, submodel_id: str, submodel: Dict[str, Any]) -> Any:
        """Overwrite a submodel wholesale.

        The only route that can remove an element the path-addressed API cannot
        reach -- a duplicate idShort makes every later `GET` and `DELETE` on
        that path answer 404, and `aas sm repair` needs a way out.
        """
        return self.put(f"/submodels/{encode_id(submodel_id)}", submodel)

    # -- submodels -------------------------------------------------------

    def list_submodels(self, limit: int = 100) -> List[Dict[str, Any]]:
        body = self.get("/submodels", params={"limit": limit})
        return _paged(body)

    def get_submodel(self, submodel_id: str) -> Dict[str, Any]:
        return self.get(f"/submodels/{encode_id(submodel_id)}")

    def create_submodel(self, submodel: Dict[str, Any]) -> Dict[str, Any]:
        return self.post("/submodels", submodel)

    def delete_submodel(self, submodel_id: str) -> None:
        self.delete(f"/submodels/{encode_id(submodel_id)}")

    # -- concept descriptions --------------------------------------------

    def list_concept_descriptions(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Every ConceptDescription the repository holds.

        Fetched wholesale rather than one per semanticId: a submodel of 130
        elements would otherwise mean 130 round trips, and the repository
        typically holds a few thousand in total. A repository that serves none
        is not an error -- indexing still works, with weaker descriptors.
        """
        try:
            return _paged(self.get("/concept-descriptions", params={"limit": limit}))
        except AasHttpError as exc:
            if exc.status == 404:
                return []
            raise

    # -- submodel elements -----------------------------------------------

    def _element_url(self, submodel_id: str, path: ElementPath) -> str:
        """Build the element URL, percent-encoding the list indices.

        ``[`` and ``]`` are gen-delims in RFC 3986 and are not legal raw in a
        path segment. BaSyx enforces it: ``Markings[0]`` is answered with a bare
        HTTP 400 and an HTML error page, while ``Markings%5B0%5D`` works. Every
        index-addressed write went through this method, so the whole
        SubmodelElementList surface was broken against a real repository -- and
        invisibly so, because the stub the tests ran against accepted either.

        ``.`` stays unescaped: it is the idShortPath separator, and AASd-002
        restricts idShorts to ``[A-Za-z][A-Za-z0-9_]*``, so nothing else in a
        rendered path needs quoting.
        """
        from urllib.parse import quote

        stem = f"/submodels/{encode_id(submodel_id)}/submodel-elements"
        return f"{stem}/{quote(path.render(), safe='._-')}" if path else stem

    def get_element(self, submodel_id: str, path: ElementPath) -> Any:
        return self.get(self._element_url(submodel_id, path))

    def get_value(self, submodel_id: str, path: ElementPath) -> Any:
        return self.get(self._element_url(submodel_id, path) + "/$value")

    def set_value(self, submodel_id: str, path: ElementPath, value: Any) -> Any:
        # PATCH /$value takes the bare value, not an element wrapper. This is
        # the one BaSyx call whose body shape differs from every other, and
        # getting it wrong yields a misleading 400.
        return self.patch(self._element_url(submodel_id, path) + "/$value", value)

    def add_element(
        self,
        submodel_id: str,
        parent: ElementPath,
        element: Dict[str, Any],
    ) -> Any:
        """POST a new element under ``parent`` (root of the submodel if empty)."""
        return self.post(self._element_url(submodel_id, parent), element)

    def replace_element(
        self, submodel_id: str, path: ElementPath, element: Dict[str, Any]
    ) -> Any:
        return self.put(self._element_url(submodel_id, path), element)

    def delete_element(self, submodel_id: str, path: ElementPath) -> None:
        self.delete(self._element_url(submodel_id, path))

    def invoke_operation(
        self, submodel_id: str, path: ElementPath, payload: Dict[str, Any]
    ) -> Any:
        return self.post(self._element_url(submodel_id, path) + "/invoke", payload)


def _paged(body: Any) -> List[Dict[str, Any]]:
    """Unwrap BaSyx's paged envelope, tolerating a bare list."""
    if body is None:
        return []
    if isinstance(body, dict):
        return body.get("result", []) or []
    if isinstance(body, list):
        return body
    return []
