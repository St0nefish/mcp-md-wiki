import io
import json
import sys
import types
import unittest
import urllib.error

# Hermes supplies agent.memory_provider; stand in for it.
agent = types.ModuleType("agent")
mp = types.ModuleType("agent.memory_provider")
mp.MemoryProvider = type("MemoryProvider", (), {})
sys.modules.setdefault("agent", agent)
sys.modules.setdefault("agent.memory_provider", mp)

from mcp_md_wiki_memory import McpMdWikiMemory  # noqa: E402


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def reply(structured, is_error=False):
    body = {"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": structured, "isError": is_error}}
    return FakeResponse(json.dumps(body).encode())


class Recorder:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        return self.response


def provider(opener, context="primary"):
    p = McpMdWikiMemory("http://kb/mcp", "secret-token", 3, opener)
    p.initialize(agent_context=context)
    return p


class ProviderTests(unittest.TestCase):
    def test_prefetch_returns_the_recall_block_and_sends_the_token_as_bearer(self):
        rec = Recorder(reply({"context": "- prefers tabs"}))
        self.assertEqual(provider(rec).prefetch("editor?"), "- prefers tabs")
        request, timeout = rec.requests[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(timeout, 3)
        sent = json.loads(request.data)
        self.assertEqual(sent["params"]["name"], "memory_recall")
        self.assertEqual(sent["params"]["arguments"], {"query": "editor?"})

    def test_prefetch_fails_open_on_every_failure(self):
        for opener in (
            Recorder(error=urllib.error.URLError("down")),
            Recorder(error=TimeoutError()),
            Recorder(error=urllib.error.HTTPError("u", 403, "no", {}, None)),
            Recorder(response=FakeResponse(b"not json")),
            Recorder(response=reply({}, is_error=True)),
        ):
            self.assertEqual(provider(opener).prefetch("q"), "")

    def test_prefetch_without_configuration_or_query_makes_no_request(self):
        rec = Recorder(reply({"context": "x"}))
        self.assertEqual(McpMdWikiMemory("", "", 3, rec).prefetch("q"), "")
        self.assertEqual(provider(rec).prefetch("   "), "")
        self.assertEqual(rec.requests, [])

    def test_only_the_primary_context_gets_or_may_use_write_tools(self):
        rec = Recorder(reply({"created": True}))
        names = lambda p: {s["name"] for s in p.get_tool_schemas()}
        self.assertIn("memory_create", names(provider(rec, "primary")))
        sub = provider(rec, "subagent")
        self.assertEqual(names(sub), {"memory_search", "memory_get"})
        refused = json.loads(sub.handle_tool_call("memory_forget", {"id": "mem-aaaaaaaaaaaa"}))
        self.assertIn("read-only", refused["error"])
        self.assertEqual(rec.requests, [], "a refused write never reaches the server")

    def test_tool_calls_pass_through_and_errors_become_json_not_exceptions(self):
        ok = provider(Recorder(reply({"returned": 0, "memories": []})))
        self.assertEqual(json.loads(ok.handle_tool_call("memory_search", {"query": "x"}))["returned"], 0)
        down = provider(Recorder(error=urllib.error.URLError("down")))
        self.assertIn("unreachable", json.loads(down.handle_tool_call("memory_search", {"query": "x"}))["error"])
        self.assertIn("unknown", json.loads(ok.handle_tool_call("shell", {}))["error"])

    def test_the_token_never_appears_in_an_error(self):
        down = provider(Recorder(error=urllib.error.URLError("secret-token in message")))
        self.assertNotIn("secret-token", down.handle_tool_call("memory_search", {"query": "x"}))


if __name__ == "__main__":
    unittest.main()
