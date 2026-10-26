#!/usr/bin/env python3
"""mocha_scan — endpoint scanner with exponential-backoff retry.

Issue an HTTP GET against a target URL and retry transient failures with
exponential backoff.  An attempt is transient when the transport raises
(URLError / timeout / connection reset) or when the server answers with a
retryable status (429 / 5xx by default).  The delay after failure k is
``base_delay * multiplier**(k-1)``, capped at ``max_delay``, optionally scaled
by full jitter (``delay * U(0, 1)``).

Stdlib only, so the tool runs in minimal CI images.  The retry engine accepts
``sleep`` / ``rng`` callables so unit tests record delays instead of waiting.

Exit codes: 0 = scan succeeded, 1 = scan failed (attempts exhausted or a
permanent HTTP error), 2 = usage error (argparse).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRYABLE_EXCEPTIONS = (urllib.error.URLError, TimeoutError, ConnectionError)


class ScanError(Exception):
    """Raised when a scan cannot be completed.

    Attributes:
        attempts: number of attempts actually performed.
        status: HTTP status code of the last response, when available.
    """

    def __init__(self, message: str, *, attempts: int, status: Optional[int] = None):
        super().__init__(message)
        self.attempts = attempts
        self.status = status


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff parameters for one scan."""

    max_attempts: int = 4
    base_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 8.0
    jitter: bool = True
    retry_statuses: frozenset = TRANSIENT_STATUSES

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")

    def delay_for(self, failed_attempt: int, rng: Callable[[], float] = random.random) -> float:
        """Delay before the retry that follows the 1-based ``failed_attempt``.

        With ``jitter`` enabled the raw exponential delay is scaled by
        ``rng()`` (full jitter); deterministic callers pass ``rng=lambda: 1.0``.
        """
        if failed_attempt < 1:
            raise ValueError("failed_attempt is 1-based")
        raw = self.base_delay * self.multiplier ** (failed_attempt - 1)
        raw = min(raw, self.max_delay)
        return raw * rng() if self.jitter else raw


def is_retryable(exc: BaseException, policy: RetryPolicy) -> bool:
    """Classify one failure as transient (retry) or permanent (abort).

    HTTPError is checked first because it subclasses URLError: a 404 must not
    be retried even though the transport type looks retryable.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in policy.retry_statuses
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


def retry_call(fn, *, policy: RetryPolicy, sleep: Callable[[float], None] = time.sleep,
               rng: Callable[[], float] = random.random, log: Optional[Callable[[str], None]] = None):
    """Run ``fn(attempt_no)`` up to ``policy.max_attempts`` times with backoff.

    Returns ``(result, attempts_used)``.  Permanent failures and an exhausted
    policy raise :class:`ScanError` chaining the last exception.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(attempt), attempt
        except Exception as exc:  # classified immediately below, then retried or re-raised
            last_exc = exc
            if not is_retryable(exc, policy):
                raise ScanError(
                    f"permanent failure after {attempt} attempt(s): {exc!r}",
                    attempts=attempt, status=getattr(exc, "code", None),
                ) from exc
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt, rng=rng)
            if log is not None:
                log(f"attempt {attempt} failed: {exc!r} — retrying in {delay:.3f}s")
            sleep(delay)
    raise ScanError(
        f"all {policy.max_attempts} attempts failed; last error: {last_exc!r}",
        attempts=policy.max_attempts, status=getattr(last_exc, "code", None),
    ) from last_exc


def scan_once(url: str, *, timeout: float = 5.0) -> Tuple[int, bytes]:
    """One GET attempt; raises on transport errors and non-2xx statuses."""
    req = urllib.request.Request(url, headers={"User-Agent": "mocha_scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def scan_url(url: str, *, policy: Optional[RetryPolicy] = None, timeout: float = 5.0,
             sleep: Callable[[float], None] = time.sleep,
             rng: Callable[[], float] = random.random,
             log: Optional[Callable[[str], None]] = None) -> Tuple[int, bytes, int]:
    """Scan ``url`` with retries; returns ``(status, body, attempts_used)``."""
    (status, body), attempts = retry_call(
        lambda _n: scan_once(url, timeout=timeout),
        policy=policy or RetryPolicy(), sleep=sleep, rng=rng, log=log,
    )
    return status, body, attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mocha_scan",
        description="Endpoint scan with exponential-backoff retry.",
    )
    parser.add_argument("url", help="target URL to GET")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--base-delay", type=float, default=0.5)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--max-delay", type=float, default=8.0)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    policy = RetryPolicy(
        max_attempts=args.max_attempts, base_delay=args.base_delay,
        multiplier=args.multiplier, max_delay=args.max_delay,
        jitter=not args.no_jitter,
    )

    def log(message: str) -> None:
        print(f"[mocha_scan] {message}", file=sys.stderr)

    try:
        status, body, attempts = scan_url(args.url, policy=policy, timeout=args.timeout, log=log)
    except ScanError as exc:
        log(f"FAIL url={args.url} attempts={exc.attempts} last_status={exc.status} error={exc}")
        return 1
    print(json.dumps({"url": args.url, "status": status, "attempts": attempts, "bytes": len(body)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
