import operator

from anytime_mesh.gated import run_tasks


def test_runs_all_tasks_and_collects_results():
    out = {}
    run_tasks(operator.add, [(i, 10) for i in range(5)], lambda a, r: out.__setitem__(a[0], r), max_procs=2,
              log_every=0)
    assert out == {i: ("ok", i + 10) for i in range(5)}


def test_errors_are_reported_not_raised():
    out = {}
    run_tasks(operator.truediv, [(1, 0)], lambda a, r: out.__setitem__(a[0], r), max_procs=1, log_every=0)
    assert out[1][0] == "error" and "ZeroDivision" in out[1][1]
