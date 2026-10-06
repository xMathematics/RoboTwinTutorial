"""projects/robotwin/metrics 的测试 —— 直接运行：``python tests/test_metrics.py``。

验证统一测评模块（成功率 / 宏平均+最差任务 / 泛化差距）：
与朴素实现交叉验证、bool 与已聚合比率两种输入同口径、并列最差任务的
取名规则、空输入显式抛 ValueError。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util as _ilu

# 本项目 metrics.py 按路径加载（唯一模块键）——仓库内 4 个项目各有 metrics.py
# （CONSTRAINTS §4.7），合并收集（仓库根 pytest / VS Code Testing）时
# sys.modules["metrics"] 会被先导入者占据，故不能裸 `import metrics`。
_mpath = Path(__file__).resolve().parents[1] / "metrics.py"
_mspec = _ilu.spec_from_file_location(f"_metrics_{_mpath.parent.name}", _mpath)
metrics = _ilu.module_from_spec(_mspec)
_mspec.loader.exec_module(metrics)

generalization_gap = metrics.generalization_gap
macro_mean_and_worst = metrics.macro_mean_and_worst
success_rate = metrics.success_rate


def test_success_rate_matches_naive():
    """(M.1) 与朴素实现（手写求和/计数）交叉验证；bool 与 0/1 同口径。"""
    rng = np.random.default_rng(7)
    for _ in range(50):
        raw = rng.integers(0, 2, size=int(rng.integers(1, 40)))
        naive = int(raw.sum()) / len(raw)
        assert abs(success_rate(raw) - naive) < 1e-12
        as_bool = [bool(v) for v in raw]
        assert abs(success_rate(as_bool) - naive) < 1e-12
        assert abs(success_rate([True, False]) - 0.5) < 1e-12


def test_macro_and_worst_match_naive():
    """(M.2) 宏平均 = 逐任务简单平均、最差 = min：与朴素实现交叉验证；
    dict 与 (序列 + 名单) 两种输入等价；最差任务名正确。"""
    per_task = {"reach": 1.0, "push": 0.55, "pick_place": 0.0}
    macro, worst, worst_name = macro_mean_and_worst(per_task)
    assert abs(macro - (1.0 + 0.55 + 0.0) / 3) < 1e-12
    assert worst == 0.0 and worst_name == "pick_place"
    macro2, worst2, name2 = macro_mean_and_worst([1.0, 0.55, 0.0],
                                                 task_names=["reach", "push", "pick_place"])
    assert (macro2, worst2, name2) == (macro, worst, worst_name)
    # 无名字输入：返回 None 而非报错。
    _m, _w, name3 = macro_mean_and_worst([0.7, 0.2])
    assert name3 is None


def test_generalization_gap_matches_naive():
    """(M.3) gap = mean(seen) − mean(unseen)：与朴素实现交叉验证；
    正值代表 unseen 变弱。"""
    rng = np.random.default_rng(11)
    for _ in range(50):
        k = int(rng.integers(1, 20))
        seen = rng.uniform(0, 1, k)
        unseen = rng.uniform(0, 1, k)
        naive = float(seen.mean() - unseen.mean())
        assert abs(generalization_gap(seen, unseen) - naive) < 1e-12
    assert abs(generalization_gap([1.0, 1.0], [0.5, 0.5]) - 0.5) < 1e-12
    # 原始 0/1 回合指示与已聚合比率同口径。
    gap_raw = generalization_gap([1, 1, 0, 0], [0, 0, 0, 0])
    gap_rates = generalization_gap([0.5], [0.0])
    assert abs(gap_raw - gap_rates) < 1e-12


def test_empty_inputs_raise():
    """空输入显式抛 ValueError（早失败优于静默返回 NaN）。"""
    for call in (
        lambda: success_rate([]),
        lambda: macro_mean_and_worst({}),
        lambda: macro_mean_and_worst([]),
        lambda: generalization_gap([], [0.5]),
        lambda: generalization_gap([0.5], []),
        lambda: generalization_gap([0.5, 0.6], [0.5]),  # 长度不一致
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"空/不一致输入未被拒绝: {call}")


def test_macro_worst_tie_breaks_to_first():
    """并列最差：取输入顺序（dict 为键序）第一个，保证报告可复现。"""
    _macro, worst, name = macro_mean_and_worst({"b": 0.3, "a": 0.3, "c": 0.9})
    assert worst == 0.3 and name == "b"
    _macro2, worst2, name2 = macro_mean_and_worst([0.3, 0.3, 0.9],
                                                  task_names=["b", "a", "c"])
    assert worst2 == 0.3 and name2 == "b"


def test_protocol_style_report_on_benchmark_records():
    """端到端小样：对 benchmark 的真实记录按协议口径出"宏平均 + 最差 +
    差距"，并核对与手工聚合一致（metrics 是 benchmark 的唯一聚合出处）。"""
    import benchmark as bm  # 局部导入避免模块加载顺序耦合
    from policies import CalibratedPolicy

    records = bm.run(CalibratedPolicy(), tasks=("reach", "push"),
                     regimes=("none",), episodes_per_task=4, seed0=0)
    table = bm.aggregate(records)
    per_task = {t: table["none"][t]["all"] for t in ("reach", "push")}
    macro, worst, worst_name = macro_mean_and_worst(per_task)
    naive_macro = sum(per_task.values()) / len(per_task)
    assert abs(macro - naive_macro) < 1e-12
    assert worst == min(per_task.values()) and worst_name == min(per_task, key=per_task.get)
    gap = generalization_gap([table["none"][t]["seen"] for t in per_task],
                             [table["none"][t]["unseen"] for t in per_task])
    naive_gap = sum(table["none"][t]["seen"] for t in per_task) / len(per_task) - \
        sum(table["none"][t]["unseen"] for t in per_task) / len(per_task)
    assert abs(gap - naive_gap) < 1e-12


if __name__ == "__main__":
    # 单点测试：python tests/test_X.py <测试名子串>；不带参数 = 全部测试。
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and pat in k]
    if not fns:
        print(f"没有匹配 '{pat}' 的测试；可用测试：")
        for k in sorted(globals()):
            if k.startswith("test_"):
                print(" ", k)
        sys.exit(1)
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)
