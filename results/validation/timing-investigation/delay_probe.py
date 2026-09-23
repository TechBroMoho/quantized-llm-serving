"""Per-chunk delivery delay d_k = client receipt - server write (same perf_counter clock)."""
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
    delays, pairs = [], []
    for _ in range(runs):
        summary, rows = asyncio.run(validation.measure_timing_accuracy())
        by_id = {o["request_id"]: o for o in summary["paired_observations"]}
        for row in rows:
            writes = by_id[row.request_id]["server_text_write_monotonic_s"]
            recv = [row.started_at + row.ttft_s]
            for g in row.itl_s: recv.append(recv[-1] + g)
            d = [r - w for r, w in zip(recv, writes)]
            delays += d
            for k in range(1, len(d)):
                if abs(d[k] - d[k-1]) > 0.001:
                    pairs.append((round(1000*d[k-1], 2), round(1000*d[k], 2)))
    for p in procs: p.terminate()
    delays.sort()
    q = lambda p: round(1000 * delays[int(p * (len(delays) - 1))], 3)
    print(json.dumps({"busy": n, "chunks": len(delays), "delay_ms": {"min": q(0), "p50": q(.5), "p90": q(.9), "p99": q(.99), "max": q(1)},
                      "negative_delays": sum(1 for x in delays if x < 0),
                      "jumps_over_1ms(prev_ms,next_ms)": pairs[:12], "jump_count": len(pairs)}))

if __name__ == "__main__":
    main()
