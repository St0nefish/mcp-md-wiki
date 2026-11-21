import asyncio
import io
import json
import os
import sys
import types
import unittest
import urllib.error

try:
    import pydantic  # noqa: F401
except ImportError:  # Open WebUI ships pydantic; stand in with a plain class.
    stub = types.ModuleType("pydantic")
    stub.BaseModel = type("BaseModel", (), {})
    sys.modules["pydantic"] = stub

import mcp_md_wiki_memory as plugin  # noqa: E402


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def reply(structured):
    body = {"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": structured}}
    return FakeResponse(json.dumps(body).encode())


class Recorder:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        return self.response


def run(filter_, body):
    return asyncio.run(filter_.inlet(body, {"id": "u1"}))


class FilterTests(unittest.TestCase):
    def setUp(self):
        os.environ["MCP_MD_WIKI_TOKEN"] = "secret-token"
        self.body = {"messages": [{"role": "user", "content": "which editor?"}]}

    def tearDown(self):
        os.environ.pop("MCP_MD_WIKI_TOKEN", None)

    def test_recall_is_prepended_as_a_system_message(self):
        rec = Recorder(reply({"context": "- prefers tabs"}))
        out = run(plugin.Filter(rec), self.body)
        self.assertEqual(out["messages"][0], {"role": "system", "content": "- prefers tabs"})
        self.assertEqual(out["messages"][1], self.body["messages"][0])
        request, timeout = rec.requests[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(timeout, 4.0)
        sent = json.loads(request.data)["params"]["arguments"]
        self.assertEqual(sent["query"], "which editor?")

    def test_every_failure_leaves_the_chat_unchanged(self):
        for opener in (
            Recorder(error=urllib.error.URLError("down")),
            Recorder(error=TimeoutError()),
            Recorder(error=urllib.error.HTTPError("u", 401, "no", {}, None)),
            Recorder(response=FakeResponse(b"junk")),
            Recorder(response=reply({"context": ""})),
        ):
            self.assertEqual(run(plugin.Filter(opener), self.body), self.body)

    def test_no_token_disabled_or_no_user_text_means_no_request(self):
        rec = Recorder(reply({"context": "x"}))
        f = plugin.Filter(rec)
        os.environ.pop("MCP_MD_WIKI_TOKEN")
        self.assertEqual(run(f, self.body), self.body)
        os.environ["MCP_MD_WIKI_TOKEN"] = "t"
        f.valves.enabled = False
        self.assertEqual(run(f, self.body), self.body)
        f.valves.enabled = True
        empty = {"messages": [{"role": "assistant", "content": "hi"}]}
        self.assertEqual(run(f, empty), empty)
        self.assertEqual(rec.requests, [])

    def test_the_token_is_not_a_valve(self):
        self.assertNotIn("token", {n.lower() for n in vars(plugin.Filter.Valves)})
        self.assertNotIn("secret-token", json.dumps(plugin.Filter().valves.__dict__))


if __name__ == "__main__":
    unittest.main()
