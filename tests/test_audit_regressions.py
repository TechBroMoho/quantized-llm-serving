"""Adversarial regression cases discovered in the pre-Phase-3 audit."""

import asyncio
import json
import time
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from llmbench.loadtest import validation
from llmbench.loadtest.client import stream_request
from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import completions_payload


def test_timing_gate_rejects_one_corrupt_request() -> None:
    original = validation.stream_request

    async def corrupt(*args, **kwargs):
        row = await original(*args, **kwargs)
        if row.request_id == "timing-0":
            row.ttft_s += 0.1
            row.e2e_s += 0.1
            row.itl_s[0] += 0.1
        return row

    with patch.object(validation, "stream_request", corrupt):
        summary, _ = asyncio.run(validation.measure_timing_accuracy())
    assert not summary["passed"], "pooled medians hide a corrupt request"


def test_open_loop_with_no_arrivals_completes_full_window() -> None:
    async def run():
        start = time.perf_counter()
        summary, rows = await run_load(
            url="http://127.0.0.1:1",
            concurrency=1,
            timeout_s=1,
            drain_s=0.01,
            payload_factory=lambda _: {},
            mode="open",
            duration_s=0.05,
            rate_per_s=0.000001,
            seed=0,
        )
        assert time.perf_counter() - start >= 0.05
        assert summary["offered_requests"] == 0
        assert rows == []

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["duplicate", "bool", "negative", "total", "no_done"])
def test_invalid_usage_or_incomplete_stream_is_rejected(fault) -> None:
    async def run():
        async def handler(_):
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            if fault == "bool":
                usage["completion_tokens"] = True
            if fault == "negative":
                usage["prompt_tokens"] = -1
            if fault == "total":
                usage["total_tokens"] = 99
            events = [{"choices": [{"text": "x"}]}, {"usage": usage}]
            if fault == "duplicate":
                events.append({"usage": usage})
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            if fault != "no_done":
                body += "data: [DONE]\n\n"
            return web.Response(text=body, content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/", handler)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            row = await stream_request(
                session,
                str(server.make_url("/")),
                completions_payload(
                    prompt="x", request_index=0, output_tokens=1, seed=0
                ),
                request_id="invalid",
            )
        assert row.status == "error", fault

    asyncio.run(run())


def test_capacity_gate_rejects_errors_even_above_rate_threshold() -> None:
    summary = {
        "measured_text_chunks_per_s": 10000,
        "client_process_cpu_cores_average": 1,
        "errored_or_cancelled_requests": 1,
        "rejected_requests": 0,
    }
    with pytest.raises(RuntimeError, match="errors or rejections"):
        validation.check_capacity(summary)


def test_load_cli_fails_after_saving_invalid_results(monkeypatch) -> None:
    from llmbench import cli

    async def invalid(_):
        return {"errored_or_cancelled_requests": 1, "rejected_requests": 0}

    monkeypatch.setattr(cli, "_run_load", invalid)
    monkeypatch.setattr(
        "sys.argv",
        ["llmbench", "load", "--url", "http://localhost", "--out", "/tmp/unused.json"],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0
