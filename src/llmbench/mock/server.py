"""A controllable OpenAI completions SSE server for local validation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

from aiohttp import web


@dataclass(frozen=True)
class MockConfig:
    """Timing and stream behavior for one mock instance."""

    ttft_s: float = 0.2
    itl_s: float = 0.02
    output_tokens: int = 3
    send_empty_chunk: bool = True
    include_usage: bool = True
    hold_open_s: float = 0.0
    timestamp_sink: Callable[[str, str, float], None] | None = None


CONFIG_KEY: web.AppKey[MockConfig] = web.AppKey("llmbench_mock_config")


def _event(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


async def _wait_for_delay(delay_s: float) -> None:
    """Wait for the configured delay without compensating for timer granularity."""
    if delay_s <= 0:
        return
    await asyncio.sleep(delay_s)


async def completion(request: web.Request) -> web.StreamResponse:
    """Emit configurable text chunks followed by a usage-only event."""
    config = request.app[CONFIG_KEY]
    body = await request.json()
    request_id = request.headers.get("X-Request-ID", "")
    prompt = body.get("prompt", "")
    prompt_tokens = (
        len(prompt) if isinstance(prompt, list) else len(str(prompt).split())
    )
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await response.prepare(request)
    try:
        if config.timestamp_sink is not None:
            config.timestamp_sink(request_id, "start", time.perf_counter())
        if config.send_empty_chunk:
            await response.write(_event({"choices": [{"text": ""}], "usage": None}))
        await _wait_for_delay(config.ttft_s)
        for index in range(config.output_tokens):
            if index:
                await _wait_for_delay(config.itl_s)
            await response.write(
                _event(
                    {
                        "choices": [{"text": "x"}],
                        "usage": None,
                    }
                )
            )
            if config.timestamp_sink is not None:
                config.timestamp_sink(request_id, "text", time.perf_counter())
        if config.include_usage:
            await response.write(
                _event(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": config.output_tokens,
                            "total_tokens": prompt_tokens + config.output_tokens,
                        },
                    }
                )
            )
        if config.hold_open_s:
            await asyncio.sleep(config.hold_open_s)
        await response.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        with suppress(ConnectionResetError):
            await response.write_eof()
    return response


async def health(_: web.Request) -> web.Response:
    """Expose a cheap readiness probe for local runners."""
    return web.json_response({"status": "ok"})


def make_app(config: MockConfig | None = None) -> web.Application:
    """Create an app whose timings can be controlled by a test."""
    app = web.Application()
    app[CONFIG_KEY] = config or MockConfig()
    app.router.add_post("/v1/completions", completion)
    app.router.add_get("/health", health)
    return app


async def serve(host: str, port: int, config: MockConfig | None = None) -> None:
    """Run the mock until interrupted."""
    runner = web.AppRunner(make_app(config))
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmbench mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ttft-ms", type=float, default=200)
    parser.add_argument("--itl-ms", type=float, default=20)
    parser.add_argument("--tokens", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(
        serve(
            args.host,
            args.port,
            MockConfig(
                ttft_s=args.ttft_ms / 1000,
                itl_s=args.itl_ms / 1000,
                output_tokens=args.tokens,
            ),
        )
    )


if __name__ == "__main__":
    main()
