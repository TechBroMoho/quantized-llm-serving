"""New gate under contention: pass rate, worst p50/p99 errors, and what the old every-gap rule would have said."""
import asyncio, json, multiprocessing, sys, time
from llmbench.loadtest import validation

def busy(stop_at):
    while time.time() < stop_at:
        pass

def main():
    n, runs = int(sys.argv[1]), int(sys.argv[2])
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=busy, args=(time.time() + 3600,), daemon=True) for _ in range(n)]
    for p in procs: p.start()
    time.sleep(1)
    fails, old_fails, worst = [], 0, {}
    for i in range(runs):
        s, _ = asyncio.run(validation.measure_timing_accuracy())
        if any(o["over_tolerance"] for o in s["paired_observations"]):
            old_fails += 1
        for m, e in s["gate"].items():
            for q in ("p50", "p99"):
                worst[f"{m}_{q}"] = max(worst.get(f"{m}_{q}", 0), e[f"{q}_relative_error"])
        if not s["passed"]:
            fails.append({m: (round(e["p50_relative_error"], 4), round(e["p99_relative_error"], 4)) for m, e in s["gate"].items() if not e["passed"]})
    for p in procs: p.terminate()
    print(json.dumps({"busy": n, "runs": runs, "new_gate_failures": len(fails), "old_every_gap_rule_failures": old_fails,
                      "worst": {k: round(v, 4) for k, v in worst.items()}, "fail_detail": fails}))

if __name__ == "__main__":
    main()
