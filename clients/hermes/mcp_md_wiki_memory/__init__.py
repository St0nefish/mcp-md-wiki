"""mcp-md-wiki memory provider for Hermes.

Recalls confirmed memories before a turn and offers the memory tools to the model.
Fails open: when the server is unreachable, slow or refuses, the agent carries on
without memory. Writes happen only for the primary agent context.

Contract: docs/memory-contract.md (version CONTRACT_VERSION below).
Configuration comes from the environment so no secret is stored in Hermes config:
  MCP_MD_WIKI_URL    base URL of the server's /mcp endpoint (required)
  MCP_MD_WIKI_TOKEN  bearer token (required)
  MCP_MD_WIKI_TIMEOUT  seconds per request, default 8
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

try:  # Hermes supplies the base class; tests stub it.
    from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover - only outside Hermes
    MemoryProvider = object  # type: ignore[assignment,misc]

CONTRACT_VERSION = 1
log = logging.getLogger(__name__)

# --- shared:begin (identical in clients/open-webui/mcp_md_wiki_memory.py) ---
WRITE_TOOLS = ("memory_create", "memory_update", "memory_forget")


def call_tool(url, token, name, arguments, timeout=8.0, opener=None):
    """Call one MCP tool over stateless Streamable HTTP and return its structured
    result (a dict). Raises on transport errors, HTTP errors and tool errors; the
    callers decide to fail open. The token is never put in an exception message."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer " + token,
        },
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise RuntimeError("memory server answered HTTP %d" % error.code) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise RuntimeError("memory server unreachable: %s" % type(error).__name__) from None
    if "error" in payload:
        raise RuntimeError("memory tool refused: %s" % payload["error"].get("message", "error"))
    result = payload.get("result", {})
    if result.get("isError"):
        raise RuntimeError("memory tool failed")
    structured = result.get("structuredContent")
    if structured is None:
        raise RuntimeError("memory server sent no structured result")
    return structured
# --- shared:end ---


class McpMdWikiMemory(MemoryProvider):
    """Memory provider backed by an mcp-md-wiki server."""

    name = "mcp-md-wiki"

    def __init__(self, url=None, token=None, timeout=None, opener=None):
        self._url = url or os.environ.get("MCP_MD_WIKI_URL", "")
        self._token = token or os.environ.get("MCP_MD_WIKI_TOKEN", "")
        self._timeout = float(timeout or os.environ.get("MCP_MD_WIKI_TIMEOUT", "8"))
        self._opener = opener
        self.agent_context = "primary"

    def is_available(self):
        return bool(self._url and self._token)

    def initialize(self, **kwargs):
        """Hermes passes the agent context; anything but `primary` is read-only."""
        self.agent_context = kwargs.get("agent_context", "primary")

    def _call(self, tool, arguments):
        return call_tool(self._url, self._token, tool, arguments, self._timeout, self._opener)

    def prefetch(self, query, **_kwargs):
        """The recall block for this turn, or "" (never raises)."""
        if not self.is_available() or not query or not query.strip():
            return ""
        try:
            return self._call("memory_recall", {"query": query}).get("context", "")
        except Exception as error:  # fail open on anything
            log.warning("memory recall skipped: %s", error)
            return ""

    def get_tool_schemas(self):
        """Search and read always; the write tools only for the primary context."""
        text = {"type": "string"}
        schemas = [
            _schema("memory_search", "Search saved memories.", {"query": text}, ["query"]),
            _schema("memory_get", "Read one memory by id.", {"id": text}, ["id"]),
        ]
        if self.agent_context == "primary":
            schemas += [
                _schema(
                    "memory_create",
                    "Save a durable memory.",
                    {"title": text, "body": text, "type": text},
                    ["title", "body", "type"],
                ),
                _schema(
                    "memory_update", "Change a memory.", {"id": text, "body": text}, ["id"]
                ),
                _schema("memory_forget", "Forget a memory.", {"id": text}, ["id"]),
            ]
        return schemas

    def handle_tool_call(self, tool_name, arguments, **_kwargs):
        """Run a memory tool for the model; the result (or a short refusal) is JSON text."""
        if tool_name in WRITE_TOOLS and self.agent_context != "primary":
            return json.dumps({"error": "memory is read-only in this context"})
        if tool_name not in ("memory_search", "memory_get") + WRITE_TOOLS:
            return json.dumps({"error": "unknown memory tool"})
        try:
            return json.dumps(self._call(tool_name, arguments or {}))
        except Exception as error:
            return json.dumps({"error": str(error)})


def _schema(name, description, properties, required):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def register(ctx):  # pragma: no cover - Hermes entry point
    ctx.register_memory_provider(McpMdWikiMemory())
