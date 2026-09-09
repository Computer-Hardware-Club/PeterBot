import asyncio
import json
import socket
from urllib.parse import urlsplit

import aiohttp
import pytest

from peterbot.tools import TOOL_SCHEMAS, ToolExecutor, _PublicResolver


def execute(name, arguments, executor=None):
    return json.loads(asyncio.run((executor or ToolExecutor("")).execute(name, json.dumps(arguments))))


@pytest.mark.parametrize(("expression", "expected"), [
    ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20), ("-(-5) + +2", 7),
    ("7 / 2", 3.5), ("-7 // 2", -4), ("7 % 3", 1), ("2 ** 10", 1024),
    ("2**-3", 0.125), ("1e100", 1e100), ("10**100", 10**100),
    ("(-3)**3", -27), ("0**0", 1), ("0.1 + 0.2", 0.30000000000000004),
])
def test_arithmetic(expression, expected):
    assert execute("calculate", {"expression": expression}) == {"status": "ok", "result": expected}


@pytest.mark.parametrize("expression", [
    "True", "False", "'hello'", "[1]", "{1: 2}", "(1,2)", "1j", "None", "x", "x.y",
    "__import__('os').system('id')", "open('/etc/passwd').read()", "sum([1,2])", "lambda: 1",
    "2 << 100", "2 & 3", "1 < 2", "1 if True else 2", "1/0", "0**-1", "(-1)**0.5",
    "10**10000", "(10**100)**100", "1e101", "1e999", "1e-100**-100", "2**(2**100)",
    "1e100 * 10", "1e100/1e-300", "1e100//1e-300", "1e100 + 1e100",
    "-" * 20 + "1", "+".join(["1"] * 30), "(" * 130 + "1" + ")" * 130,
    "1\n+2", "1\x00", "1 # comment\n", "", " " * 10,
])
def test_arithmetic_rejects_code_nonfinite_and_resource_abuse(expression):
    assert execute("calculate", {"expression": expression})["status"] == "error"


@pytest.mark.parametrize(("name", "arguments"), [
    ("unknown", {}), ("calculate", {}), ("calculate", {"expression": "1", "extra": 3}),
    ("calculate", {"expression": 1}), ("calculate", {"expression": True}),
    ("web_search", {"query": "hello", "url": "http://localhost"}),
    ("web_search", {"query": "x" * 301}), ("web_search", {"query": "a\tb"}),
    ("web_search", {"query": "hello !google"}), ("web_search", {"query": "!!ddg hello"}),
    ("web_search", {"query": ":news hello"}), ("web_search", {"query": "hello :en"}),
    ("fetch_public_page", {"url": "https://www.python.org", "headers": {"Authorization": "secret"}}),
])
def test_argument_validation(name, arguments):
    assert execute(name, arguments)["status"] == "error"


@pytest.mark.parametrize("arguments", [
    "[]", "null", "NaN", "{", '{"expression":"1","expression":"2"}',
    "[" * 1500 + "]" * 1500, " " * 5000,
])
def test_invalid_raw_json_is_contained(arguments):
    result = json.loads(asyncio.run(ToolExecutor("").execute("calculate", arguments)))
    assert result["status"] == "error"


def test_schemas_require_only_declared_arguments():
    assert {tool["function"]["name"] for tool in TOOL_SCHEMAS} == {"calculate", "web_search", "fetch_public_page"}
    for tool in TOOL_SCHEMAS:
        schema = tool["function"]["parameters"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


class FakeContent:
    def __init__(self, body, *, error=None):
        self.body = body
        self.error = error
        self.read_bytes = 0

    async def iter_chunked(self, size):
        if self.error:
            raise self.error
        for offset in range(0, len(self.body), size):
            chunk = self.body[offset:offset + size]
            self.read_bytes += len(chunk)
            yield chunk


class FakeResponse:
    def __init__(self, body=b"", *, status=200, headers=None, content_length=None, error=None):
        self.status = status
        self.headers = headers or {"Content-Type": "text/html"}
        self.content_length = content_length
        self.charset = None
        self.content = FakeContent(body, error=error)
        self.before_enter = None

    async def __aenter__(self):
        if self.before_enter:
            await self.before_enter()
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    def __init__(self, responses, **kwargs):
        self.responses = responses
        self.kwargs = kwargs
        self.requests = []
        self.closed = False

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self):
        self.closed = True
        if self.kwargs.get("connector"):
            await self.kwargs["connector"].close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
        return False


def search_executor(data=None, response=None):
    executor = ToolExecutor("http://100.64.0.1:8888")
    executor.http_session = FakeSession([response or FakeResponse(json.dumps(data).encode())])
    return executor


def test_search_uses_fixed_endpoint_without_redirects():
    executor = search_executor({"results": []})
    assert execute("web_search", {"query": "site:python.org asyncio"}, executor)["status"] == "ok"
    assert executor.http_session.requests == [(
        "http://100.64.0.1:8888/search",
        {"params": {"q": "site:python.org asyncio", "format": "json", "safesearch": 1}, "allow_redirects": False},
    )]
    asyncio.run(executor.close())
    assert executor.http_session.closed


def test_search_disabled():
    assert execute("web_search", {"query": "hello"})["status"] == "unavailable"


BLOCKED_URLS = [
    "http://localhost/", "http://localhost./", "http://printer/", "http://a.local/",
    "http://a.internal/", "http://a.home.arpa/", "http://a.invalid/", "http://a.example.com/",
    "http://127.0.0.1/", "http://10.1.2.3/", "http://172.16.0.1/", "http://192.168.1.1/",
    "http://169.254.169.254/", "http://100.100.100.100/", "http://0.0.0.0/",
    "http://224.0.0.1/", "http://240.0.0.1/", "http://192.0.2.1/",
    "http://[::1]/", "http://[fc00::1]/", "http://[fe80::1]/", "http://[ff02::1]/",
    "http://[::ffff:127.0.0.1]/", "http://[2002:7f00:1::]/", "http://[fe80::1%25en0]/",
    "http://2130706433/", "http://127.1/", "http://0x7f000001/", "http://0177.0.0.1/",
    "http://0x7f.0.0.0x01/", "http://good.org@localhost/", "https://user:pass@python.org/",
    "http://python.org:99999/", "http://python.org:bad/", "file:///etc/passwd", "ftp://python.org/",
    "javascript:alert(1)", "https://python.org\\@localhost/", "https://python.org/a b",
    "https://python.org/\nprivate", "http://[not-ip]/", "http://%31%32%37.0.0.1/",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_search_filters_nonpublic_or_malformed_links(url):
    executor = search_executor({"results": [{"title": "bad", "url": url, "content": "bad"}]})
    result = execute("web_search", {"query": "hello"}, executor)
    assert result["status"] == "partial"
    assert result["results"] == []


def test_search_sanitizes_html_and_bounds_fields_results_and_output():
    items = [{"title": "<b>Python</b>\x00", "url": "https://www.python.org/", "content": "<script>secret()</script><p>A &amp; B</p><style>hide</style><p>C</p>"}]
    items += [{"title": '"' * 1000, "url": "https://www.wikipedia.org/", "content": '\\"' * 1000}] * 10
    executor = search_executor({"results": items, "unresponsive_engines": [["engine", "timeout"]]})
    output = asyncio.run(executor.execute("web_search", '{"query":"hello"}'))
    result = json.loads(output)
    assert result["status"] == "partial"
    assert len(output) <= 8000
    assert len(result["results"]) <= 5
    assert result["results"][0] == {"title": "Python", "url": "https://www.python.org/", "snippet": "A & B C"}
    assert all(len(item["title"]) <= 160 and len(item["snippet"]) <= 700 for item in result["results"])


@pytest.mark.parametrize("response", [
    FakeResponse(status=302, headers={"Location": "http://localhost"}),
    FakeResponse(status=500), FakeResponse(b"not json"), FakeResponse(b'[]'),
    FakeResponse(b'{"results":{}}'), FakeResponse(b"x", content_length=262145),
    FakeResponse(b"x" * 300000), FakeResponse(b"x", headers={"Content-Encoding": "gzip"}),
    FakeResponse(error=aiohttp.ClientError("secret server detail")),
    FakeResponse(error=asyncio.TimeoutError("secret detail")),
])
def test_search_failures_are_bounded_and_do_not_leak_details(response):
    executor = search_executor(response=response)
    result = execute("web_search", {"query": "hello"}, executor)
    assert result["status"] == "unavailable"
    assert "secret" not in str(result)
    assert response.content.read_bytes <= 262144 + 8192
    assert len(executor.http_session.requests) == 1


@pytest.mark.parametrize("origin", [
    "file:///tmp", "http://u:p@host", "http://host/path", "http://host?token=secret",
    "http://host#x", "http://host\n", "http://host:99999",
])
def test_invalid_search_configuration_is_rejected_without_echoing_secrets(origin):
    with pytest.raises(ValueError) as error:
        ToolExecutor(origin)
    assert "secret" not in str(error.value)


def fake_fetch(monkeypatch, responses, dns_addresses=None):
    sessions = []
    dns_calls = []

    async def getaddrinfo(host, port, **kwargs):
        dns_calls.append((host, port))
        addresses = dns_addresses or ["93.184.216.34"]
        return [(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port)) for address in addresses]

    def factory(**kwargs):
        session = FakeSession(responses, **kwargs)
        original_get = session.get

        def get(url, **get_kwargs):
            response = original_get(url, **get_kwargs)

            async def resolve():
                host = urlsplit(url).hostname
                await kwargs["connector"]._resolve_host(host, urlsplit(url).port or 443)

            response.before_enter = resolve
            return response

        session.get = get
        sessions.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", lambda self, *args, **kwargs: getaddrinfo(*args, **kwargs))
    return sessions, dns_calls


@pytest.mark.parametrize("url", BLOCKED_URLS + ["http://python.org:22", "https://python.org:8443"])
def test_fetch_rejects_nonpublic_url_before_network(monkeypatch, url):
    sessions, calls = fake_fetch(monkeypatch, [])
    result = execute("fetch_public_page", {"url": url})
    assert result["status"] == "error"
    assert not sessions and not calls


def test_fetch_reads_main_text_with_safe_session(monkeypatch):
    sessions, calls = fake_fetch(monkeypatch, [FakeResponse(
        b"<html><head><title>Title</title></head><nav>links</nav><main><h1>Hello</h1><p>World &amp; club</p><script>attack()</script></main><footer>footer</footer></html>"
    )])
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result == {"status": "ok", "url": "https://www.python.org/", "text": "Hello World & club"}
    session = sessions[0]
    assert session.requests[0][1] == {"allow_redirects": False}
    assert session.kwargs["trust_env"] is False
    assert session.kwargs["auto_decompress"] is False
    assert isinstance(session.kwargs["cookie_jar"], aiohttp.DummyCookieJar)
    assert isinstance(session.kwargs["connector"]._resolver, _PublicResolver)
    assert session.kwargs["connector"]._use_dns_cache is False
    assert session.kwargs["connector"]._ssl is True
    assert "Authorization" not in session.kwargs["headers"]
    assert session.closed
    assert calls == [("www.python.org", 443)]


@pytest.mark.parametrize("addresses", [
    ["127.0.0.1"], ["93.184.216.34", "10.0.0.1"], ["100.73.210.66"], ["169.254.169.254"],
    ["224.0.0.1"], ["::1"], ["fe80::1"], ["fc00::1"], ["93.184.216.34", "::ffff:127.0.0.1"],
])
def test_dns_all_answers_are_checked_and_private_answers_block_connection(monkeypatch, addresses):
    sessions, calls = fake_fetch(monkeypatch, [FakeResponse(b"hello")], dns_addresses=addresses)
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["status"] == "unavailable"
    assert sessions[0].responses == []
    assert len(calls) == 1
    assert sessions[0].closed


def test_resolver_pins_numeric_addresses_and_does_not_resolve_twice(monkeypatch):
    async def check():
        calls = []
        async def getaddrinfo(host, port, **kwargs):
            calls.append(host)
            addresses = ["93.184.216.34"] if len(calls) == 1 else ["127.0.0.1"]
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port)) for ip in addresses]
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
        async with aiohttp.TCPConnector(resolver=_PublicResolver(), use_dns_cache=False) as connector:
            result = await connector._resolve_host("www.python.org", 443)
        assert result[0]["host"] == "93.184.216.34"
        assert result[0]["hostname"] == "www.python.org"
        assert result[0]["flags"] & socket.AI_NUMERICHOST
        assert calls == ["www.python.org"]
    asyncio.run(check())


def test_fetch_validates_redirect_and_never_requests_private_target(monkeypatch):
    sessions, _ = fake_fetch(monkeypatch, [FakeResponse(status=302, headers={"Location": "http://169.254.169.254/latest"})])
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["status"] == "error"
    assert len(sessions) == 1


def test_fetch_relative_redirect_gets_new_connection_and_dns(monkeypatch):
    sessions, calls = fake_fetch(monkeypatch, [
        FakeResponse(status=302, headers={"Location": "/docs"}),
        FakeResponse(b"2 < 3 and 5 > 4", headers={"Content-Type": "text/plain"}),
    ])
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["text"] == "2 < 3 and 5 > 4"
    assert result["url"] == "https://www.python.org/docs"
    assert len(sessions) == 2 and len(calls) == 2
    assert all(session.closed for session in sessions)


def test_fetch_stops_after_three_redirects(monkeypatch):
    sessions, _ = fake_fetch(monkeypatch, [FakeResponse(status=302, headers={"Location": "/next"}) for _ in range(5)])
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["status"] == "unavailable"
    assert len(sessions) == 4


@pytest.mark.parametrize("response", [
    FakeResponse(status=403), FakeResponse(status=302),
    FakeResponse(b"pdf", headers={"Content-Type": "application/pdf"}),
    FakeResponse(b"binary", headers={"Content-Type": "application/octet-stream"}),
    FakeResponse(b"gzip", headers={"Content-Type": "text/html", "Content-Encoding": "gzip"}),
    FakeResponse(b"x", content_length=262145), FakeResponse(b"x" * 300000),
    FakeResponse(error=aiohttp.ClientError("secret credentials")),
    FakeResponse(error=asyncio.TimeoutError("secret")),
    FakeResponse(b"<script>nothing readable</script>"),
])
def test_fetch_failures_are_bounded(monkeypatch, response):
    sessions, _ = fake_fetch(monkeypatch, [response])
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["status"] == "unavailable"
    assert "secret" not in str(result)
    assert response.content.read_bytes <= 262144 + 8192
    assert sessions[0].closed


def test_fetch_output_is_bounded_including_json_escaping(monkeypatch):
    fake_fetch(monkeypatch, [FakeResponse(('\\"' * 5000).encode(), headers={"Content-Type": "text/plain"})])
    output = asyncio.run(ToolExecutor("").execute("fetch_public_page", '{"url":"https://www.python.org/"}'))
    assert len(output) <= 6000
    assert json.loads(output)["status"] == "partial"


def test_search_session_is_separate_and_has_no_auth_proxy_or_cookie_state(monkeypatch):
    sessions = []
    def factory(**kwargs):
        session = FakeSession([FakeResponse(b'{"results":[]}')], **kwargs)
        sessions.append(session)
        return session
    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    executor = ToolExecutor("http://100.64.0.1:8888")
    assert execute("web_search", {"query": "hello"}, executor)["status"] == "ok"
    assert sessions[0].kwargs["trust_env"] is False
    assert isinstance(sessions[0].kwargs["cookie_jar"], aiohttp.DummyCookieJar)
    assert sessions[0].kwargs["headers"] == {"Accept": "application/json", "Accept-Encoding": "identity"}
    asyncio.run(executor.close())
    assert sessions[0].closed


def test_fetch_dns_timeout_is_inside_total_deadline(monkeypatch):
    sessions, _ = fake_fetch(monkeypatch, [FakeResponse(b"hello")])
    async def slow_dns(self, *args, **kwargs):
        await asyncio.sleep(1)
        raise AssertionError("DNS should have been cancelled")
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", slow_dns)
    executor = ToolExecutor("", timeout_seconds=0.005)
    result = execute("fetch_public_page", {"url": "https://www.python.org/"}, executor)
    assert result["status"] == "unavailable"
    assert sessions[0].closed


def test_fetch_redirect_to_private_dns_is_rechecked(monkeypatch):
    sessions, _ = fake_fetch(monkeypatch, [
        FakeResponse(status=302, headers={"Location": "https://evil.org/"}),
        FakeResponse(b"private service data"),
    ])
    async def dns(self, host, port, **kwargs):
        address = "93.184.216.34" if host == "www.python.org" else "10.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", dns)
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["status"] == "unavailable"
    assert "private service data" not in str(result)
    assert len(sessions) == 2 and all(session.closed for session in sessions)


def test_fetch_cancellation_propagates_and_closes_session(monkeypatch):
    sessions, _ = fake_fetch(monkeypatch, [FakeResponse(error=asyncio.CancelledError())])
    with pytest.raises(asyncio.CancelledError):
        execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert sessions[0].closed



def test_fetch_skips_large_decorative_art_before_useful_club_content(monkeypatch):
    art = "+#@%.- " * 1600
    html = (
        "<main><h1>Computer Hardware Club</h1><pre>" + art + "</pre>"
        "<p>We're a student-run club at Oregon State University.</p>"
        "<p>Hands-on workshops. No experience required.</p></main>"
    )
    fake_fetch(monkeypatch, [FakeResponse(html.encode())])
    result = execute("fetch_public_page", {"url": "https://computerhardwareclub.org/index.html"})
    assert result["status"] == "ok"
    assert "student-run club" in result["text"]
    assert "Hands-on workshops" in result["text"]
    assert "+#@%" not in result["text"]
    assert len(result["text"]) < 200


def test_fetch_preserves_prose_short_symbols_and_explicit_code(monkeypatch):
    code = "+++++[>+++++<-]>." * 20
    prose = "        Hardware workshops welcome students. " * 10
    html = (
        "<main><p>" + prose + "</p><p>+++ --- ***</p>"
        "<pre><code>" + code.replace("<", "&lt;").replace(">", "&gt;") + "</code></pre>"
        "<pre>" + "for item in items: print(item)\n" * 10 + "</pre></main>"
    )
    fake_fetch(monkeypatch, [FakeResponse(html.encode())])
    result = execute("fetch_public_page", {"url": "https://www.python.org/"})
    assert result["status"] == "ok"
    assert "Hardware workshops welcome students." in result["text"]
    assert "+++ --- ***" in result["text"]
    assert code in result["text"]
    assert "for item in items: print(item)" in result["text"]



def test_fetch_ignores_accessibility_hidden_art_split_across_nodes(monkeypatch):
    html = (
        '<main><div aria-hidden="true"><div>'
        + '<span>+ # @ % .</span>' * 600
        + '</div><br><span>still hidden</span></div>'
        + "<p>We're a student-run club.</p></main>"
    )
    fake_fetch(monkeypatch, [FakeResponse(html.encode())])
    result = execute("fetch_public_page", {"url": "https://computerhardwareclub.org/index.html"})
    assert result["text"] == "We're a student-run club."
