"""projects/robotwin/policies 的测试 —— 直接运行：``python tests/test_policies.py``。

验证两种脚本对照策略的行为主张（即教程 §4.2.3/§6.4 的剂量论断在本
迷你基准里的复现）：
- calibrated 在标定世界（none 档）三任务成功率 ≥ 0.9；
- robust 在 none 档同样保持高位；
- DR 剂量 ↑ → calibrated 宏平均单调下降、且 strong 档显著低于 robust；
- robust 的每回合自标定确实把连杆长/半径测准（相对真值误差远小于
  名义模型的偏差）。

门限均为 seed0=0、N=40 回合/格的实测值再留 ≥0.1 裕量（详见各行注释）。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark as bm  # noqa: E402
import dr  # noqa: E402
import tasks as tk  # noqa: E402
from policies import CalibratedPolicy, RobustPolicy  # noqa: E402

N_EPISODES = 40  # 与 demo.py 一致（seen 20 + unseen 20）


def _run_macro(policy_cls, regimes):
    """跑协议并返回 {档位: (每任务成功率 dict, 宏平均)}。"""
    records = bm.run(policy_cls(), tasks=tk.TASKS, regimes=regimes,
                     episodes_per_task=N_EPISODES, seed0=0)
    table = bm.aggregate(records)
    out = {}
    for regime in regimes:
        per_task = {t: table[regime][t]["all"] for t in tk.TASKS}
        macro, _worst, _name = bm.mt.macro_mean_and_worst(per_task)
        out[regime] = (per_task, macro)
    return out


def test_calibrated_none_three_tasks_at_least_090():
    """calibrated@none 三任务 ≥ 0.9（实测 1.00/1.00/1.00——标定即完美，
    门限留 0.1 裕量给跨 numpy 版本的边界翻转）。"""
    per_task, macro_val = _run_macro(CalibratedPolicy, ("none",))["none"]
    for task, rate in per_task.items():
        assert rate >= 0.9, (task, rate)
    print("\n[calibrated@none] " + ", ".join(f"{t}={r:.2f}" for t, r in per_task.items()))


def test_robust_none_stays_high():
    """robust@none 三任务 ≥ 0.85（实测 1.00/0.95/1.00；自标定在无噪声的
    none 档退化为一次多余测量，不伤害成功率）。"""
    per_task, macro_val = _run_macro(RobustPolicy, ("none",))["none"]
    for task, rate in per_task.items():
        assert rate >= 0.85, (task, rate)
    print("\n[robust@none] " + ", ".join(f"{t}={r:.2f}" for t, r in per_task.items()))


def test_strong_calibrated_below_robust():
    """strong 档对照（本项目的核心主张，对应教程 §6.4 Table 3 的
    "干净 vs 随机化"两列）：
    - calibrated 宏平均 ≤ 0.65（实测 0.48）；
    - robust 宏平均 ≥ 0.70（实测 0.82）；
    - 两者之差 ≥ 0.15（实测 0.33）。"""
    cal = _run_macro(CalibratedPolicy, ("strong",))["strong"][1]
    rob = _run_macro(RobustPolicy, ("strong",))["strong"][1]
    assert cal <= 0.65, cal
    assert rob >= 0.70, rob
    assert rob - cal >= 0.15, (cal, rob)
    print("\n[strong] calibrated macro={cal:.2f} < robust macro={rob:.2f}"
          f" (gap {rob - cal:.2f})")


def test_calibrated_dose_monotone_decrease():
    """剂量曲线（教程 §4.2.3/§6.4 的"DR 剂量 vs 成功率"）：
    none ≥ 0.95（实测 1.00）；mild ≤ none − 0.05（实测 0.85）；
    strong ≤ mild − 0.15（实测 0.48）——单调下降。"""
    macros = {reg: m for reg, (_p, m) in _run_macro(CalibratedPolicy, dr.REGIMES).items()}
    assert macros["none"] >= 0.95, macros
    assert macros["mild"] <= macros["none"] - 0.05, macros
    assert macros["strong"] <= macros["mild"] - 0.15, macros
    print("\n[calibrated dose curve] " + ", ".join(
        f"{r}={macros[r]:.2f}" for r in dr.REGIMES))


def test_robust_measurement_tracks_true_lengths():
    """robust 的每回合自标定把世界测准：strong 档 8 个回合上，
    每次测量连杆长相对真值的误差 < 0.06 m（实测 ~1e-2 量级），且总误差
    显著小于名义模型的偏差（strong 档臂长漂移最多 ±15% ≈ 0.07 m）。"""
    worst_measured = 0.0
    worst_nominal = 0.0
    sum_measured = 0.0
    sum_nominal = 0.0
    for episode in range(8):
        params = dr.sample("strong", np.random.default_rng([0, 0, 0, 0, episode]))
        env = tk.make_task("reach", params)
        obs = env.reset(seed=100 + episode)
        policy = RobustPolicy()
        policy.reset("reach")
        for step in range(4):  # 前 4 步是探测测量
            action = policy.act(obs, step)
            obs, _r, _d, _i = env.step(action)
        for side in (0, 1):
            true_links = np.array(params.left_link if side == 0 else params.right_link)
            measured = np.array(policy.measured_links[side])
            nominal = np.array(dr.NOMINAL_LINK_LEFT if side == 0 else dr.NOMINAL_LINK_RIGHT)
            err_m = float(np.max(np.abs(measured - true_links)))
            err_n = float(np.max(np.abs(nominal - true_links)))
            worst_measured = max(worst_measured, err_m)
            worst_nominal = max(worst_nominal, err_n)
            sum_measured += err_m
            sum_nominal += err_n
            # 每次测量都准（< 0.06 m）；个别回合的臂长可能恰好接近名义值，
            # "测量优于名义"因此在总口径上断言（总误差至少省一半）。
            assert err_m < 0.06, (episode, side, err_m)
        # 半径读数（带噪均值）也应贴近真值。
        assert abs(policy.measured_radius - params.object_radius) < 0.02
    assert sum_measured < 0.5 * sum_nominal, (sum_measured, sum_nominal)
    print(f"\n[robust-measure] worst |measured-true| = {worst_measured:.4f} m, "
          f"worst |nominal-true| = {worst_nominal:.4f} m, "
          f"sum(measured)/sum(nominal) = {sum_measured / sum_nominal:.2f}")


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
