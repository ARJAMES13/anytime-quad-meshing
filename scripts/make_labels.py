"""Mesh every part of a split with the given methods and append one JSONL row per (part, method).

Resumable: rows already in --out are skipped (crash/timeout/out-of-memory rows are retried). Each task runs in its
own process, started only while enough RAM is free. im_matched / im_k3 need qqs_default rows first (they read its
budget n*); qqs_pert<f> needs the first-pass count of qqs_default. Run qqs_default first, then the rest.

Examples:
  python scripts/make_labels.py --part test --methods qqs_default --out results/labels/my_test.jsonl
  python scripts/make_labels.py --part test --methods im_matched,im_k3,gmsh_blossom --out results/labels/my_test.jsonl
"""
import argparse
import json
import os

from _common import rel

RERUN_STATUSES = {"crash", "timeout", "infra_error"}
METHODS = ("qqs_default, qqs_uniform, qqs_curv_hi, qqs_feat, qqs_curv_feat, qqs_pert<f>, "
           "im_default, im_matched, im_k3, gmsh_blossom")


def task(tid, solid_id, path, method, extra):
    from anytime_mesh.labelling import evaluate
    return evaluate(solid_id, path, method, extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=rel("configs/data.toml"), help="dataset roots (TOML)")
    ap.add_argument("--split", default=rel("configs/splits/v1.json"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test", "test_ood"])
    ap.add_argument("--methods", required=True, help="comma-separated: " + METHODS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N parts (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2, help="max parallel worker processes")
    ap.add_argument("--timeout", type=float, default=900, help="seconds per (part, method)")
    a = ap.parse_args()

    from anytime_mesh.data import load_config, resolve
    from anytime_mesh.gated import run_tasks
    cfg = load_config(a.data_config)
    split = json.load(open(a.split))
    ids = split[a.part][a.offset:]
    if a.limit:
        ids = ids[:a.limit]
    methods = a.methods.split(",")
    kept, done = [], set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if r.get("status") in RERUN_STATUSES:
                continue
            kept.append(line.rstrip("\n") + "\n")
            done.add((r["solid_id"], r["method"]))
        with open(a.out, "w") as fh:          # drop rows that will be retried
            fh.writelines(kept)
    n1_of, nstar_of = {}, {}
    for line in kept:
        r = json.loads(line)
        if r["method"] == "qqs_default" and "tp" in r and r["tp"]["p1"].get("valid"):
            n1_of[r["solid_id"]] = r["tp"]["p1"]["n_quads"]
            if r["tp"]["p2"].get("valid"):
                nstar_of[r["solid_id"]] = r["tp"]["p2"]["n_quads"]
    needs = lambda sid, m: (sid in n1_of) if m.startswith("qqs_pert") else (sid in nstar_of) if m in ("im_matched", "im_k3") else True
    extra = lambda sid, m: ({"n1": n1_of[sid]} if m.startswith("qqs_pert") else
                            {"n_star": nstar_of[sid]} if m in ("im_matched", "im_k3") else None)
    todo = [(sid, m) for sid in ids for m in methods if (sid, m) not in done and needs(sid, m)]
    print(f"{len(todo)} (part, method) tasks to run ({len(done)} done)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fh = open(a.out, "a")

    def on_result(args, res):
        _, sid, _, m = args[:4]
        if res[0] == "ok":
            row = res[1]
        else:
            st = "timeout" if "timeout" in str(res[1]) else "crash"
            row = {"solid_id": sid, "method": m, "status": st, "valid": False, "score": 10.0}
        row.pop("msg", None)
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    tasks = [(f"{sid}|{m}", sid, resolve(sid, cfg), m, extra(sid, m)) for sid, m in todo]
    run_tasks(task, tasks, on_result, max_procs=a.procs, timeout_s=a.timeout)
    fh.close()


if __name__ == "__main__":
    main()
