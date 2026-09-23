"""Serialized and static-batched Hugging Face completions, streamed per token."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.streamers import BaseStreamer


@dataclass
class Job:
    prompt: list[int]
    max_new_tokens: int
    events: asyncio.Queue[int | BaseException | None] = field(
        default_factory=asyncio.Queue
    )
    count: int = 0


class TokenStreamer(BaseStreamer):
    """Pass each generated token and batch row directly to the event loop."""

    def __init__(self, jobs: list[Job], loop: asyncio.AbstractEventLoop):
        self.jobs = jobs
        self.loop = loop
        self.prompt_seen = False

    def put(self, value: torch.Tensor) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if value.ndim == 1:
            value = value.unsqueeze(1)
        if value.shape != (len(self.jobs), 1):
            raise ValueError(f"unexpected generated-token shape: {tuple(value.shape)}")
        for job, token in zip(self.jobs, value[:, 0].tolist(), strict=True):
            self.loop.call_soon_threadsafe(job.events.put_nowait, int(token))

    def end(self) -> None:
        for job in self.jobs:
            self.loop.call_soon_threadsafe(job.events.put_nowait, None)


class Baseline:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        mode: str,
        batch_size: int = 4,
        batch_wait_ms: float = 10,
        min_new_tokens_override: int | None = None,
    ) -> None:
        if mode not in {"naive", "static"}:
            raise ValueError("mode must be naive or static")
        if batch_size < 1 or batch_wait_ms < 0:
            raise ValueError("invalid batch settings")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.mode = mode
        self.batch_size = 1 if mode == "naive" else batch_size
        self.batch_wait_s = 0 if mode == "naive" else batch_wait_ms / 1000
        self.min_new_tokens_override = min_new_tokens_override
        self.pending: asyncio.Queue[Job] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        self.generated_batch_sizes: list[int] = []

    async def start(self) -> None:
        self.worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker

    async def _run(self) -> None:
        while True:
            first = await self.pending.get()
            jobs = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_wait_s
            while len(jobs) < self.batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self.pending.get(), remaining)
                except TimeoutError:
                    break
                if candidate.max_new_tokens != first.max_new_tokens:
                    # A single generate call has one length setting.
                    self.pending.put_nowait(candidate)
                    break
                jobs.append(candidate)
            try:
                self.generated_batch_sizes.append(len(jobs))
                await asyncio.to_thread(
                    self._generate, jobs, asyncio.get_running_loop()
                )
            except Exception as exc:
                for job in jobs:
                    job.events.put_nowait(exc)
                    job.events.put_nowait(None)

    def _generate(self, jobs: list[Job], loop: asyncio.AbstractEventLoop) -> None:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer needs pad or EOS token")
        width = max(len(job.prompt) for job in jobs)
        ids = [[pad_id] * (width - len(job.prompt)) + job.prompt for job in jobs]
        masks = [
            [0] * (width - len(job.prompt)) + [1] * len(job.prompt) for job in jobs
        ]
        device = self.model.device
        generation = GenerationConfig(  # type: ignore[no-untyped-call]
            max_new_tokens=jobs[0].max_new_tokens,
            min_new_tokens=(
                jobs[0].max_new_tokens
                if self.min_new_tokens_override is None
                else self.min_new_tokens_override
            ),
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            diversity_penalty=0.0,
            length_penalty=1.0,
            early_stopping=False,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=False,
            output_scores=False,
            forced_eos_token_id=None,
            forced_bos_token_id=None,
            suppress_tokens=None,
            bad_words_ids=None,
        )
        with torch.inference_mode():
            self.model.generate(
                input_ids=torch.tensor(ids, dtype=torch.long, device=device),
                attention_mask=torch.tensor(masks, dtype=torch.long, device=device),
                generation_config=generation,
                streamer=TokenStreamer(jobs, loop),
            )


def _event(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def make_app(baseline: Baseline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await baseline.start()
        try:
            yield
        finally:
            await baseline.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completion(body: dict[str, Any]) -> StreamingResponse:
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            prompt = baseline.tokenizer.encode(prompt, add_special_tokens=False)
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in prompt
            )
        ):
            raise HTTPException(400, "prompt must be nonempty text or token IDs")
        maximum = body.get("max_tokens")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise HTTPException(400, "max_tokens must be a positive integer")
        required: dict[str, Any] = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        if any(body.get(key) != value for key, value in required.items()):
            raise HTTPException(
                400, "only explicit greedy streaming settings supported"
            )
        if body.get("stop") not in (None, []):
            raise HTTPException(400, "stop sequences are unsupported")
        job = Job(prompt=prompt, max_new_tokens=maximum)
        await baseline.pending.put(job)

        async def stream() -> Any:
            while True:
                item = await job.events.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    yield _event({"error": str(item)})
                    break
                job.count += 1
                text = baseline.tokenizer.decode([item], skip_special_tokens=False)
                yield _event({"choices": [{"text": text}], "usage": None})
            yield _event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(job.prompt),
                        "completion_tokens": job.count,
                        "total_tokens": len(job.prompt) + job.count,
                    },
                }
            )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face baseline server")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--mode", choices=("naive", "static"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # type: ignore[no-untyped-call]
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.cuda()
    baseline = Baseline(
        model,
        tokenizer,
        mode=args.mode,
        batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    uvicorn.run(make_app(baseline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
