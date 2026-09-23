"""Run the timing validation repeatedly and record which checks fail, with GC pauses."""
import asyncio, gc, json, multiprocessing, sys, time

from llmbench.loadtest import validation

def busy(stop_at):
    x = 0
    while time.time() < stop_at:
        x += 1

def main():
    cond = sys.argv[1]
    runs = int(sys.argv[2])
    procs = []
    if cond.startswith("heap"):
        import torch, transformers  # noqa: F401
        import transformers.models.gpt2.modeling_gpt2  # noqa
        junk = [dict(a=i, b=[i]) for i in range(3_000_000)]  # simulate a big live heap
        print("gc objects", len(gc.get_objects()), file=sys.stderr)
    if cond.endswith("cpu"):
        n = int(sys.argv[3]) if len(sys.argv) > 3 else multiprocessing.cpu_count()
        stop = time.time() + 3600
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=busy, args=(stop,), daemon=True) for _ in range(n)]
        for p in procs: p.start()
        time.sleep(1)
    pauses = []
    state = {}
    def cb(phase, info):
        if phase == "start":
            state["t"] = time.perf_counter()
        else:
            pauses.append((info["generation"], time.perf_counter() - state["t"]))
    gc.callbacks.append(cb)
    out = []
    for i in range(runs):
        pauses.clear()
        summary, _ = asyncio.run(validation.measure_timing_accuracy())
        worst = {}
        for obs in summary["paired_observations"]:
            for k, v in obs["relative_errors"].items():
                worst[k] = max(worst.get(k, 0), v)
        bad = {k: round(v, 4) for k, v in worst.items() if v > 0.05}
        out.append({"passed": summary["passed"], "bad": bad,
                    "ttft_med_err": summary["ttft_relative_error"], "itl_med_err": summary["itl_relative_error"],
                    "max_gc_ms": round(1000*max((p for _, p in pauses), default=0), 2),
                    "gen2": sum(1 for g, _ in pauses if g == 2)})
    for p in procs: p.terminate()
    fails = [o for o in out if not o["passed"]]
    print(json.dumps({"cond": cond, "runs": runs, "failures": len(fails), "fail_detail": fails[:10],
                      "max_gc_ms_all": max(o["max_gc_ms"] for o in out)}, indent=1))

if __name__ == "__main__":
    main()
