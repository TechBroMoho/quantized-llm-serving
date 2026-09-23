"""pytest plugin: log GC pauses and test time spans (diagnosis only)."""
import gc, json, os, time

LOG = os.environ["GCWATCH_LOG"]
_events = []
_start = {}

def _cb(phase, info):
    if phase == "start":
        _start["t"] = time.perf_counter()
    else:
        d = time.perf_counter() - _start["t"]
        if d > 0.0005:
            _events.append({"kind": "gc", "gen": info["generation"], "at": _start["t"], "ms": round(d * 1000, 2), "collected": info["collected"]})

gc.callbacks.append(_cb)

def pytest_runtest_call(item):
    _events.append({"kind": "start", "test": item.nodeid, "at": time.perf_counter(), "gc_objects": len(gc.get_objects()), "threads": __import__("threading").active_count()})

def pytest_runtest_logreport(report):
    if report.when == "call":
        _events.append({"kind": "end", "test": report.nodeid, "at": time.perf_counter(), "outcome": report.outcome})

def pytest_sessionfinish(session):
    with open(LOG, "w") as f:
        json.dump(_events, f)
