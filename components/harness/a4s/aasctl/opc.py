"""OPC UA as a transport for AAS Operations, and as a surface of its own.

Some assets do not offer HTTP. A Siemens S7-1500 offers an OPC UA server, and
what an AAS calls an Operation is, on that server, a write to a node in a data
block. The Festo CP Lab is two of those plus an open62541 server on a
Revolution Pi, and none of them has an endpoint `httpx` can post to.

**One delegation mechanism, two transports.** The existing shells write
``invocationDelegation = http://host:port/path``. An OPC UA target is a server
*and* a node, so it is written::

    opc.tcp://192.0.2.10:4042#ns=3;s="ApplicationModule_DB"."xStart"

One string, in the qualifier that already exists, with the node in the fragment
where a fragment belongs. `aas op ls`, the approval gate and the safety
declaration all keep working unchanged; only the invoker learns a second
scheme. Inventing a parallel `opcDelegation` qualifier would have meant
teaching every one of them about it.

**Reading is not invoking.** A pressure is read, not commanded, and modelling
it as an Operation would put every measurement through the approval gate. So
readable nodes are Properties carrying a ``valueSource`` qualifier and are
fetched with ``aas opc read``; only actions are Operations.

**Types are read from the server, not guessed from the text.** ``true`` is a
Boolean on one node and the string ``"true"`` on another, and a value of the
wrong variant is refused by the server with an error that names the type but
not the node. The node's own ``DataType`` decides, which costs one extra read
and removes the entire class of mistake.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, unquote

# The qualifier that says where a readable value comes from. `services_
# delegated` answers "how do I act"; this answers "how do I look", and the two
# are different questions about the same server.
VALUE_SOURCE = "valueSource"

DEFAULT_TIMEOUT = 10.0

# How deep `browse` walks by default. A PLC address space is large -- an
# S7-1500 exposes every symbol in every data block -- and an unbounded walk
# returns tens of thousands of nodes into an agent's context window.
DEFAULT_DEPTH = 2
MAX_NODES = 500


class OpcError(RuntimeError):
    """The server could not be reached, or the request cannot be made as asked."""


def is_opc_url(url: Optional[str]) -> bool:
    return bool(url) and str(url).lower().startswith("opc.tcp://")


def split_target(url: str) -> Tuple[str, Optional[str]]:
    """Separate ``opc.tcp://host:port`` from the node id in its fragment.

    The node id is *not* unquoted blindly. An S7 node id contains quotes and
    dots -- ``ns=3;s="ApplicationModule_DB"."xStart"`` -- and percent-decoding
    a string that was never encoded turns a legitimate ``%`` in a symbol name
    into a decoding error. Only a fragment that actually looks encoded is
    decoded.
    """
    parts = urlsplit(url)
    endpoint = f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")
    node = parts.fragment or None
    if node and "%" in node and "=" not in node.split("%", 1)[0][-3:]:
        node = unquote(node)
    return endpoint, node


def tag_name(node_id: Optional[str]) -> str:
    """The PLC's own name for a node, short enough to sit beside a label.

    An operator reading `ConveyorForward` on a screen and `xQA1_A1` in TIA
    Portal has to be able to see that they are the same signal. The semantic
    name says what it means; this says what to search for.

    `ns=3;s="ApplicationModule_DB"."xReady"` -> `ApplicationModule_DB.xReady`
    `ns=3;s="xQA1_A1"`                       -> `xQA1_A1`
    `ns=1;s=Pressure_1`                      -> `Pressure_1`
    """
    if not node_id:
        return ""
    text = str(node_id)
    # Everything after the identifier marker; numeric ids have no name to give.
    for marker in (";s=", ";i="):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    return text.replace('"', "").strip()


@dataclass
class NodeValue:
    node_id: str
    value: Any
    data_type: str
    writable: bool
    status: str = "Good"
    # Filled by `write`: what the node held before, and every value seen in the
    # window after. Empty for a plain read.
    before: Any = None
    wanted: Any = None
    observed: List[Any] = field(default_factory=list)

    def line(self) -> str:
        access = "rw" if self.writable else "r-"
        return f"{access} {self.node_id:<48} = {self.value!r} ({self.data_type})"

    def settled(self) -> str:
        """What the write actually did, in one word the caller can branch on.

        `stood`      -- the value is still there. A level signal, set.
        `consumed`   -- it was seen set and then cleared. A trigger the PLC took.
        `not_seen`   -- never observed set, and unchanged from before.
        `unknown`    -- no observations, or nothing that fits.

        **`not_seen` is not the same as "ignored", and this cannot tell them
        apart.** A PLC scan is milliseconds and the fastest observation here is
        a network round trip, so a trigger consumed within one scan is
        invisible at this sampling rate and looks exactly like a write nothing
        acted on. The only reliable evidence is the *effect* -- `Busy` rising,
        a position bit changing -- which is a different node and a different
        read. Saying more than that would be the same mistake as the first
        version, which reported every disappearance as a healthy trigger.
        """
        if not self.observed:
            return "unknown"
        if self.observed[-1] == self.wanted:
            return "stood"
        if any(v == self.wanted for v in self.observed):
            return "consumed"
        if self.before == self.observed[-1]:
            return "not_seen"
        return "unknown"

    def explain(self) -> str:
        return {
            "stood": "the value is standing",
            "consumed": "the value was set and then cleared by the program, "
                        "which is what a trigger bit does when it is taken",
            "not_seen": "the value never appeared in the readings and the node "
                        "is unchanged. Either the program consumed it within a "
                        "scan -- too fast to observe over the network -- or "
                        "nothing acted on it. Judge by the effect: watch Busy, "
                        "or the position bit, rather than this one.",
            "unknown": "the outcome could not be determined from the readings",
        }[self.settled()]


@dataclass
class BrowsedNode:
    node_id: str
    browse_name: str
    node_class: str
    depth: int
    children: List["BrowsedNode"] = field(default_factory=list)

    def line(self) -> str:
        return f"{'  ' * self.depth}{self.browse_name:<40} {self.node_class:<12} {self.node_id}"


# ---------------------------------------------------------------------------
# the asyncua calls, each one self-contained
# ---------------------------------------------------------------------------

def _run(coroutine: Any) -> Any:
    """Run one coroutine, on Windows with the selector loop asyncua needs.

    A new event loop per call rather than a persistent one. A CLI verb is one
    request and then the process exits; holding a session open across verbs
    would mean holding a subscription an operator cannot see, on a machine
    that may be running.
    """
    try:
        from asyncua import Client  # noqa: F401
    except ImportError as exc:  # pragma: no cover - install-time failure
        raise OpcError(
            "OPC UA support needs the `opc` extra: pip install -e '.[opc]'"
        ) from exc

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coroutine)


def _connect(endpoint: str, timeout: float) -> Any:
    from asyncua import Client

    return Client(url=endpoint, timeout=timeout)


async def _variant_of(node: Any) -> Any:
    """The node's own VariantType, which is what a write has to match."""
    return await node.read_data_type_as_variant_type()


def _coerce(text: str, variant: Any) -> Any:
    """Turn a command-line string into the variant the node declares.

    Deliberately strict about booleans. `bool("false")` is `True` in Python,
    and a start command that silently becomes `True` because the operator
    typed `false` is the worst possible failure of this function.
    """
    from asyncua import ua

    name = variant.name
    if name == "Boolean":
        lowered = text.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise OpcError(
            f"{text!r} is not a boolean. Write true or false -- this node is a "
            "Boolean, and guessing at anything else is how a machine gets started."
        )
    if name in ("SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32",
                "Int64", "UInt64"):
        try:
            return int(text)
        except ValueError as exc:
            raise OpcError(f"{text!r} is not an integer; this node is a {name}") from exc
    if name in ("Float", "Double"):
        try:
            return float(text)
        except ValueError as exc:
            raise OpcError(f"{text!r} is not a number; this node is a {name}") from exc
    if name == "String":
        return text
    if name == "DateTime":
        from datetime import datetime

        return datetime.fromisoformat(text)
    # Anything else is passed through and the server decides. Refusing here
    # would block node types this tool has not met yet for no gain.
    del ua
    return text


async def _read_one(client: Any, node_id: str) -> NodeValue:
    from asyncua import ua

    node = client.get_node(node_id)
    value = await node.read_value()
    variant = await _variant_of(node)
    try:
        access = await node.read_attribute(ua.AttributeIds.AccessLevel)
        writable = bool(int(access.Value.Value) & 0x02)
    except Exception:  # noqa: BLE001 - AccessLevel is optional on some servers
        writable = False
    return NodeValue(node_id=node_id, value=value, data_type=variant.name,
                     writable=writable)


async def _browse_one(node: Any, depth: int, max_depth: int,
                      budget: List[int]) -> List[BrowsedNode]:
    if depth > max_depth or budget[0] <= 0:
        return []
    out: List[BrowsedNode] = []
    for child in await node.get_children():
        if budget[0] <= 0:
            break
        budget[0] -= 1
        try:
            name = (await child.read_browse_name()).Name
            node_class = (await child.read_node_class()).name
        except Exception as exc:  # noqa: BLE001 - one unreadable node is not the walk
            name, node_class = f"<{type(exc).__name__}>", "Unknown"
        entry = BrowsedNode(node_id=child.nodeid.to_string(), browse_name=name,
                            node_class=node_class, depth=depth)
        entry.children = await _browse_one(child, depth + 1, max_depth, budget)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# what the CLI calls
# ---------------------------------------------------------------------------

def read_evidence(endpoint: str, node_ids: List[str], *, timeout: float = DEFAULT_TIMEOUT):
    """Quality and source/server timestamps for monitoring; strictly read-only."""
    async def go():
        async with _connect(endpoint, timeout) as client:
            rows = []
            for node_id in node_ids:
                data = await client.get_node(node_id).read_data_value(raise_on_bad_status=False)
                rows.append({'value': data.Value.Value,
                             'dataType': data.Value.VariantType.name,
                             'status': 'Good' if data.StatusCode.is_good() else str(data.StatusCode),
                             'source_timestamp': data.SourceTimestamp.isoformat() if data.SourceTimestamp else None,
                             'server_timestamp': data.ServerTimestamp.isoformat() if data.ServerTimestamp else None})
            return rows
    async def bounded():
        return await asyncio.wait_for(go(), timeout=timeout * 3)
    try:
        return _run(bounded())
    except Exception as exc:
        raise OpcError(f'OPC evidence unavailable: {type(exc).__name__}') from exc


def read(endpoint: str, node_ids: List[str], *,
         timeout: float = DEFAULT_TIMEOUT) -> List[NodeValue]:
    """Read one or more nodes in a single session."""

    async def go() -> List[NodeValue]:
        async with _connect(endpoint, timeout) as client:
            return [await _read_one(client, n) for n in node_ids]

    try:
        return _run(go())
    except OpcError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced with the endpoint named
        raise OpcError(f"{endpoint}: {type(exc).__name__}: {exc}") from exc


# How long to watch a node after writing it, and how often. Short enough not
# to hold a session, long enough to cover a PLC scan and the round trip.
SETTLE_WINDOW = 1.2
SETTLE_INTERVAL = 0.15


def write(endpoint: str, node_id: str, value: str, *,
          timeout: float = DEFAULT_TIMEOUT,
          expect_type: Optional[str] = None,
          settle: float = SETTLE_WINDOW) -> NodeValue:
    """Write one node, coercing to the type the node declares.

    **The value is watched for a moment afterwards, not read once.** A single
    read taken immediately cannot tell three different things apart, and they
    need different responses:

    * the PLC consumed a trigger -- the bit stood and was cleared by the
      program, which is success;
    * the write never took -- the bit was false before, false after, and the
      module ignored it;
    * the write stands -- a level signal such as a mode bit, still set.

    Reading once and reporting "read back: False" collapsed the first two into
    one sentence. That sentence was in the console as "normal for a trigger bit
    the PLC clears in the same scan", and it was reassuring through four start
    attempts on a module that was in manual mode and ignoring every one of
    them. The returned `NodeValue` now carries the whole series, and
    `settled()` says which of the three happened.
    """

    async def go() -> NodeValue:
        import time

        from asyncua import ua

        async with _connect(endpoint, timeout) as client:
            node = client.get_node(node_id)
            variant = await _variant_of(node)
            if expect_type and variant.name.lower() != expect_type.lower():
                raise OpcError(
                    f"{node_id} is a {variant.name}, not a {expect_type}. "
                    "The type was stated and does not match, so nothing was written."
                )
            wanted = _coerce(value, variant)
            before = await node.read_value()
            await node.write_value(ua.DataValue(ua.Variant(wanted, variant)))

            observed: List[Any] = []
            deadline = time.monotonic() + max(settle, 0.0)
            while True:
                observed.append(await node.read_value())
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(SETTLE_INTERVAL)

            result = await _read_one(client, node_id)
            result.before = before
            result.observed = observed
            result.wanted = wanted
            return result

    try:
        return _run(go())
    except OpcError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpcError(f"{endpoint}: {type(exc).__name__}: {exc}") from exc


def browse(endpoint: str, node_id: Optional[str] = None, *,
           depth: int = DEFAULT_DEPTH,
           timeout: float = DEFAULT_TIMEOUT) -> List[BrowsedNode]:
    """Walk the address space from a node, bounded in depth and count."""

    async def go() -> List[BrowsedNode]:
        async with _connect(endpoint, timeout) as client:
            root = client.get_node(node_id) if node_id else client.nodes.objects
            return await _browse_one(root, 0, depth, [MAX_NODES])

    try:
        return _run(go())
    except OpcError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpcError(f"{endpoint}: {type(exc).__name__}: {exc}") from exc


def call(endpoint: str, node_id: str, arguments: List[str], *,
         timeout: float = DEFAULT_TIMEOUT) -> Any:
    """Call an OPC UA method, coercing arguments to its declared signature.

    Separate from `write` because they are different things on the wire, and
    because a server that offers methods says so: the S7-1500 in the Festo cell
    offers none, and everything it can be asked to do is a write.
    """

    async def go() -> Any:
        async with _connect(endpoint, timeout) as client:
            method = client.get_node(node_id)
            parent = await method.get_parent()
            if parent is None:
                raise OpcError(f"{node_id} has no parent object to call it on")
            try:
                signature, _ = await _method_signature(method)
            except Exception:  # noqa: BLE001 - a method without metadata still calls
                signature = []
            if signature and len(signature) != len(arguments):
                raise OpcError(
                    f"{node_id} takes {len(signature)} arguments "
                    f"({', '.join(a.Name for a in signature)}), {len(arguments)} given"
                )
            coerced = [
                _coerce(raw, _variant_from_datatype(arg.DataType))
                for raw, arg in zip(arguments, signature)
            ] if signature else list(arguments)
            return await parent.call_method(method, *coerced)

    try:
        return _run(go())
    except OpcError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpcError(f"{endpoint}: {type(exc).__name__}: {exc}") from exc


async def _method_signature(method: Any) -> Tuple[List[Any], List[Any]]:
    from asyncua import ua

    inputs: List[Any] = []
    outputs: List[Any] = []
    for child in await method.get_children():
        name = (await child.read_browse_name()).Name
        if name == "InputArguments":
            inputs = await child.read_value() or []
        elif name == "OutputArguments":
            outputs = await child.read_value() or []
    del ua
    return inputs, outputs


def _variant_from_datatype(data_type: Any) -> Any:
    """Map a DataType NodeId onto a VariantType, falling back to String."""
    from asyncua import ua

    try:
        return ua.VariantType(data_type.Identifier)
    except Exception:  # noqa: BLE001 - a custom type is passed through as text
        return ua.VariantType.String


def endpoints(endpoint: str, *, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """What the server says about itself: product, version, namespaces."""

    async def go() -> Dict[str, Any]:
        async with _connect(endpoint, timeout) as client:
            info: Dict[str, Any] = {"endpoint": endpoint}
            try:
                build = await client.nodes.server.get_child(
                    ["0:ServerStatus", "0:BuildInfo"]
                )
                value = await build.read_value()
                info.update(
                    productName=value.ProductName,
                    manufacturerName=value.ManufacturerName,
                    softwareVersion=value.SoftwareVersion,
                    buildNumber=value.BuildNumber,
                )
            except Exception as exc:  # noqa: BLE001 - BuildInfo is optional
                info["buildInfo"] = f"unavailable ({type(exc).__name__})"
            info["namespaces"] = await client.get_namespace_array()
            return info

    try:
        return _run(go())
    except OpcError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpcError(f"{endpoint}: {type(exc).__name__}: {exc}") from exc
