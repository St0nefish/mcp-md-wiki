"""
title: mcp-md-wiki memory
author: mcp-md-wiki
version: 1.0.0
description: Adds confirmed memories from an mcp-md-wiki server to each chat turn. Fails open.

The bearer token is read from the environment (MCP_MD_WIKI_TOKEN), never from a
Valve, so it is not stored in the Open WebUI database. Contract:
docs/memory-contract.md (version CONTRACT_VERSION below).
"""

import asyncio
import json
import os
import urllib.error
import urllib.request

from pydantic import BaseModel

CONTRACT_VERSION = 1

# --- shared:begin (identical in clients/hermes/mcp_md_wiki_memory/__init__.py) ---
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


class Filter:
    class Valves(BaseModel):
        url: str = "http://mcp-md-wiki:8080/mcp"
        timeout_seconds: float = 4.0
        max_items: int = 8
        max_chars: int = 1500
        enabled: bool = True

    def __init__(self, opener=None):
        self.valves = self.Valves()
        self._opener = opener

    async def inlet(self, body: dict, __user__: dict = None, **_kwargs) -> dict:
        """Prepend the recall block as a system message. Any failure leaves the chat
        untouched: memory is an aid, never a dependency."""
        token = os.environ.get("MCP_MD_WIKI_TOKEN", "")
        if not self.valves.enabled or not token:
            return body
        messages = body.get("messages") or []
        query = next(
            (
                m.get("content")
                for m in reversed(messages)
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ),
            "",
        )
        if not query.strip():
            return body
        try:
            result = await asyncio.to_thread(
                call_tool,
                self.valves.url,
                token,
                "memory_recall",
                {
                    "query": query,
                    "max_items": self.valves.max_items,
                    "max_chars": self.valves.max_chars,
                },
                self.valves.timeout_seconds,
                self._opener,
            )
        except Exception:
            return body
        context = result.get("context", "")
        if not context:
            return body
        return {
            **body,
            "messages": [{"role": "system", "content": context}] + list(messages),
        }

    async def outlet(self, body: dict, __user__: dict = None, **_kwargs) -> dict:
        return body
