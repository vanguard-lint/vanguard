"""Tests for tools/mocha_scan.py — stdlib only.

Run:  python3 tools/test_mocha_scan.py -v
"""
import http.server
import json
import subprocess
import sys
import threading
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mocha_scan as ms


def policy(**kw):
    base = dict(max_attempts=4, base_delay=0.5, multiplier=2.0, max_delay=8.0, jitter=False)
    base.update(kw)
    return ms.RetryPolicy(**base)


class RetryEngineTests(unittest.TestCase):
    def test_success_first_attempt_no_sleep(self):
        calls, sleeps = [], []

        def fn(_n):
            calls.append(_n)
            return "ok"

        res, attempts = ms.retry_call(fn, policy=policy(), sleep=sleeps.append, rng=lambda: 1.0)
        self.assertEqual(res, "ok")
        self.assertEqual(attempts, 1)
        self.assertEqual(calls, [1])
        self.assertEqual(sleeps, [])

    def test_transient_failures_then_success_exact_backoff(self):
        sleeps, calls = [], []

        def fn(_n):
            calls.append(_n)
            if _n < 3:
                raise urllib.error.URLError("conn refused")
            return "pong"

        res, attempts = ms.retry_call(fn, policy=policy(), sleep=sleeps.append, rng=lambda: 1.0)
        self.assertEqual((res, attempts), ("pong", 3))
        self.assertEqual(sleeps, [0.5, 1.0])  # base * multiplier**(k-1)

    def test_backoff_capped_at_max_delay(self):
        sleeps = []

        def fn(_n):
            if _n < 5:
                raise TimeoutError("slow")
            return "ok"

        pol = policy(max_attempts=5, base_delay=0.5, multiplier=2.0, max_delay=1.2)
        _, attempts = ms.retry_call(fn, policy=pol, sleep=sleeps.append, rng=lambda: 1.0)
        self.assertEqual(attempts, 5)
        self.assertEqual(sleeps, [0.5, 1.0, 1.2, 1.2])

    def test_http_404_is_permanent_no_retry(self):
        sleeps = []

        def fn(_n):
            raise urllib.error.HTTPError("http://x", 404, "nf", {}, None)

        with self.assertRaises(ms.ScanError) as cm:
            ms.retry_call(fn, policy=policy(), sleep=sleeps.append, rng=lambda: 1.0)
        self.assertEqual(cm.exception.attempts, 1)
        self.assertEqual(cm.exception.status, 404)
        self.assertEqual(sleeps, [])

    def test_transient_status_codes_are_retried(self):
        for code in (429, 500, 502, 503, 504):
            with self.subTest(code=code):
                sleeps = []

                def fn(_n, c=code):
                    if _n == 1:
                        raise urllib.error.HTTPError("http://x", c, "t", {}, None)
                    return "ok"

                _, attempts = ms.retry_call(fn, policy=policy(), sleep=sleeps.append, rng=lambda: 1.0)
                self.assertEqual(attempts, 2)
                self.assertEqual(sleeps, [0.5])

    def test_exhaustion_raises_scan_error(self):
        sleeps = []

        def fn(_n):
            raise ConnectionError("reset")

        with self.assertRaises(ms.ScanError) as cm:
            ms.retry_call(fn, policy=policy(max_attempts=3), sleep=sleeps.append, rng=lambda: 1.0)
        self.assertEqual(cm.exception.attempts, 3)
        self.assertEqual(len(sleeps), 2)

    def test_jitter_scales_delay(self):
        p = policy(jitter=True)
        self.assertEqual(p.delay_for(1, rng=lambda: 0.5), 0.25)
        self.assertEqual(p.delay_for(3, rng=lambda: 1.0), 2.0)

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            policy(max_attempts=0)
        with self.assertRaises(ValueError):
            policy(base_delay=-1)
        with self.assertRaises(ValueError):
            policy(multiplier=0.5)
        with self.assertRaises(ValueError):
            policy(base_delay=2.0, max_delay=1.0)
        with self.assertRaises(ValueError):
            ms.RetryPolicy().delay_for(0)


class FlakyHandler(http.server.BaseHTTPRequestHandler):
    hits = 0
    fail_first = 2

    def do_GET(self):
        FlakyHandler.hits += 1
        if FlakyHandler.hits <= FlakyHandler.fail_first:
            self.send_response(503)
            self.end_headers()
        else:
            body = b"pong"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):  # keep test output clean
        pass


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FlakyHandler.hits = 0
        FlakyHandler.fail_first = 2
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_scan_retries_503_then_succeeds(self):
        FlakyHandler.hits = 0
        FlakyHandler.fail_first = 2
        pol = policy(max_attempts=4, base_delay=0.01, multiplier=2.0, jitter=False)
        status, body, attempts = ms.scan_url(f"http://127.0.0.1:{self.port}/", policy=pol, timeout=2.0)
        self.assertEqual((status, body, attempts), (200, b"pong", 3))

    def test_scan_success_first_attempt_no_retry(self):
        FlakyHandler.hits = 0
        FlakyHandler.fail_first = 0
        pol = policy(max_attempts=4, base_delay=0.01, jitter=False)
        status, body, attempts = ms.scan_url(f"http://127.0.0.1:{self.port}/", policy=pol, timeout=2.0)
        self.assertEqual((status, body, attempts), (200, b"pong", 1))

    def test_cli_end_to_end(self):
        FlakyHandler.hits = 0
        FlakyHandler.fail_first = 2
        out = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "mocha_scan.py"),
             f"http://127.0.0.1:{self.port}/", "--max-attempts", "4",
             "--base-delay", "0.01", "--no-jitter"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        payload = json.loads(out.stdout)
        self.assertEqual(payload["status"], 200)
        self.assertEqual(payload["attempts"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
