# Progress

## Status

Phase 1 acceptance passed. Phase 2 has not started. No GPU or Modal function
has run. This session is limited to local, $0 work.

## Phase log

- **Phase 0 (2026-09-23):** Copied the instruction file to `AGENTS.md` and
  made sync an explicit rule. Verified GitHub login as `TechBroMoho` and the
  presence of Modal secret `huggingface` by read-only checks. Created the
  Python 3.12 project layout and documented the model estimate and measurement
  rules. `make setup` resolved 44 packages and checked 43. `make check` passed:
  Ruff reported "All checks passed!" and 3 files formatted; mypy reported
  "Success: no issues found in 2 source files"; pytest reported "2 passed";
  `cmp CLAUDE.md AGENTS.md` succeeded. `uv run llmbench` displayed its help,
  and `uv run modal --version` reported 1.3.3. No GPU stack or model checkpoint
  has been exercised.
- **Phase 1 (2026-09-23):** Implemented the controllable OpenAI completions
  SSE mock; closed-loop and Poisson open-loop load generation; chunk-based
  TTFT/ITL/TPOT/E2E metrics; usage-based token counts; bounded drain
  accounting; CLI and raw result writers. Initial timing validation passed at median
  TTFT 0.201355 s and ITL 0.020251 s for expected 0.200 s / 0.020 s
  (±5%). The intentional empty-chunk timer mutation failed as expected with
  99.6% TTFT error; it was restored. Capacity validation sustained 41,960.7
  text chunks/s for 30 seconds at 256 streams (6.99× the 6,000/s threshold,
  35,960.7/s above it), with 0.98 average client CPU cores, no errors, and no
  rejected requests. `make check`: 11 tests passed; Ruff and strict mypy
  passed. `make mock-validate` passed. A separate CLI smoke run completed
  four requests with zero errors. The measurements in this first log entry were
  superseded by the timing audit immediately below.
- **Phase 1 timing audit (2026-09-23):** Replaced the mock's final-millisecond
  busy spin (which compensated for coarse asyncio timer granularity) with plain
  `asyncio.sleep`. Accuracy now compares client timings to same-host monotonic
  timestamps recorded at handler start and immediately after each text write
  returns; requested 200/20 ms delays are not the reference. Five-sample client
  medians were TTFT 0.202324 s vs server 0.201397 s (0.46% difference) and ITL
  0.021327 s vs server 0.021258 s (0.33%), within ±5%. Treating the initial
  empty event as text failed with 99.7% TTFT difference; production code was
  restored and the corrected acceptance passed. Capacity recheck sustained
  40,405.8 chunks/s (6.73× threshold, 34,405.8/s margin) at 0.98 client CPU
  cores, zero errors/rejections; 256 requests remained at the measurement
  deadline and completed during the drain, so they are separately reported as
  late and excluded from in-window throughput. Full `make check` passed (11
  tests, Ruff, mypy, instruction-file comparison). No Modal job ran.

## Spend log

| Date | Phase | Activity | GPU | Seconds | Estimate | Running total |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2026-09-23 | 0 | Local setup and read-only account checks | None | 0 | $0.00 | $0.00 |
| 2026-09-23 | 1 | Mock, tests, timing and capacity validation, CLI smoke | None | 0 | $0.00 | $0.00 |

The account's remaining Modal credit has not been verified. No billable
Modal work has been requested in this session.

## Things that went wrong

- The local sandbox could not reach GitHub, Modal, or PyPI. Read-only account
  checks and the local dependency install passed with network access enabled.
  No remote compute was started.
- The sandbox denied localhost socket binding. The local integration and
  acceptance checks ran with localhost access; a first capacity attempt also
  exposed a wrong health-probe path. The probe was fixed and the full
  validation then passed.
