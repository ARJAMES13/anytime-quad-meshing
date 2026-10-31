"""Memory-gated parallel task runner (laptop-safe).

Each task runs in its own spawned child process at BelowNormal priority. A new task starts only while
free RAM >= START_FREE_GB and memory use <= START_MAX_PCT. If free RAM falls below FLOOR_FREE_GB the
most recently started task is killed and re-queued (it will be retried later). Results come back through
a queue; `on_result(args, result)` is called in the parent for every finished task.
"""
import multiprocessing as mp
import os
import time

import psutil

START_FREE_GB = 2.5
START_MAX_PCT = 90.0
FLOOR_FREE_GB = 1.0


def _child(fn, args, conn):
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass
    try:
        res = (args[0], "ok", fn(*args))
    except Exception as e:  # noqa: BLE001
        res = (args[0], "error", repr(e)[:300])
    conn.send(res)          # private pipe: killing another task can never corrupt this channel
    conn.close()


def _mem_ok():
    v = psutil.virtual_memory()
    return v.available / 2**30 >= START_FREE_GB and v.percent <= START_MAX_PCT


def run_tasks(fn, tasks, on_result, max_procs=8, timeout_s=900, log_every=10, max_requeue=2):
    """tasks: list of argument tuples (first element = task id). fn must be importable (top-level)."""
    from multiprocessing.connection import wait as mp_wait
    ctx = mp.get_context("spawn")
    pending = list(tasks)
    requeues = {}
    running = {}            # tid -> (process, start_time, args)
    done = 0
    last_log = 0
    t0 = time.time()
    while pending or running:
        # start new work while memory allows
        while pending and len(running) < max_procs and _mem_ok():
            args = pending.pop(0)
            r_conn, w_conn = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(fn, args, w_conn), daemon=True)
            p.start()
            w_conn.close()
            running[args[0]] = (p, time.time(), args, r_conn)
            time.sleep(2.0)             # let it allocate before judging memory again
        # memory floor: kill the youngest task and requeue it
        if running and psutil.virtual_memory().available / 2**30 < FLOOR_FREE_GB:
            tid = max(running, key=lambda k: running[k][1])
            p, _, args, rc = running.pop(tid)
            p.kill(); p.join(5); rc.close()
            requeues[tid] = requeues.get(tid, 0) + 1
            if requeues[tid] <= max_requeue:
                pending.append(args)
            else:
                on_result(args, ("error", "memory"))
            continue
        conns = {v[3]: k for k, v in running.items()}
        for rc in mp_wait(list(conns), timeout=1.0) if conns else []:
            tid = conns[rc]
            try:
                _, status, res = rc.recv()
            except (EOFError, OSError):
                continue            # child died without sending: handled by the liveness check below
            p, _, args, _ = running.pop(tid)
            p.join(10); rc.close()
            on_result(args, (status, res))
            done += 1
        if not conns:
            time.sleep(1.0)
        # timeouts and silent deaths (e.g. C++ std::terminate inside gmsh)
        now = time.time()
        for tid in list(running):
            p, st, args, rc = running[tid]
            if now - st > timeout_s or (not p.is_alive() and p.exitcode not in (None, 0)):
                if p.is_alive():
                    p.kill()
                p.join(5)
                running.pop(tid); rc.close()
                on_result(args, ("error", "timeout" if now - st > timeout_s else f"died exit={p.exitcode}"))
                done += 1
        if log_every and done - last_log >= log_every:
            last_log = done
            v = psutil.virtual_memory()
            print(f"[gated] done {done}/{len(tasks)} running {len(running)} pending {len(pending)} "
                  f"free {v.available / 2**30:.1f}GB {time.time() - t0:.0f}s", flush=True)
    return done
