"""projects/control_planning/cbf.py 的测试 —— 直接运行：``python tests/test_cbf.py``。

验证教程第 07 章的 CBF-QP 安全滤波教学实现（Ames et al., ECC 2019）：
屏障 h 与李导数 (7.7) 的一致性、无滤波撞入 / 有滤波保持 h ≥ -tol 的对照
（(7.4)/(7.5) 连续条件的离散容差标注）、闭式解 (7.13) 的投影性与"最小侵入"、
多约束投影迭代（(L_gh L_ghᵀ + εI) 可逆的批量闭式 + POCS 收尾）。
全部确定性，无随机性。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbf import (  # noqa: E402
    affine_fields,
    barrier_affine,
    cbf_filter,
    project_halfspaces,
    simulate_filtered,
)

#: 对照场景（教程 07.2 的动机场景）：直冲障碍的恒定加速度指令。
CENTER = np.array([2.4, 0.0])
RADIUS = 0.5
X0 = np.array([0.0, 0.0, 1.5, 0.0])
U_DES = np.array([1.2, 0.0])
DT = 1e-3
N_STEPS = 6000
#: 离散容差：(7.4) 是连续时间条件；欧拉离散下线性被控对象精确、唯一余量是
#: h 的曲率项 O(dt²·ḧ)。本场景 approach 朝向障碍中心（n 恒定），实测最小
#: h = -5.4e-16 —— 门限 -1e-6 之余量覆盖曲率效应（详见 DEBUG.md §3 C-2）。


def test_barrier_affine_matches_central_difference():
    """h 的解析梯度（经 L_f h / L_g h）与中心差分一致。"""
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = np.concatenate([rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)])
        if np.linalg.norm(x[:2] - CENTER) < RADIUS + 0.2:
            continue  # 跳过障碍内部（n 无定义方向）
        h0, l_fh, l_gh = barrier_affine(x, CENTER, RADIUS, kappa=0.4)
        f, g = affine_fields(x)
        # L_f h = dh/dt·(1/ḋt) 方向导数：沿漂移场 f 的一阶差分。
        eps = 1e-6
        h_f = barrier_affine(x + eps * f, CENTER, RADIUS, kappa=0.4)[0]
        assert abs((h_f - h0) / eps - l_fh) < 1e-4
        # L_g h 各列：沿输入场方向的导数 = κnᵀ 的对应分量。
        for j in range(2):
            h_g = barrier_affine(x + eps * g[:, j], CENTER, RADIUS, kappa=0.4)[0]
            assert abs((h_g - h0) / eps - l_gh[j]) < 1e-4
            assert abs(l_gh[j] - 0.4 * (x[:2] - CENTER)[j]
                       / np.linalg.norm(x[:2] - CENTER)) < 1e-9


def test_unfiltered_desired_control_collides():
    """无滤波：直冲指令必须撞入（h 变负、净距为负）——安全滤波的必要性。"""
    raw = simulate_filtered(X0, U_DES, [(CENTER, RADIUS)], DT, N_STEPS, filtered=False)
    assert raw.min_h < 0.0
    assert raw.min_dist < 0.0
    print(f"\n[raw] min h {raw.min_h:.3f} m, min net distance {raw.min_dist:.3f} m")


def test_filtered_keeps_h_above_tolerance():
    """有滤波：全程 h ≥ -1e-6（离散容差见文件头注）、无穿透、投影残余为机器精度。"""
    safe = simulate_filtered(X0, U_DES, [(CENTER, RADIUS)], DT, N_STEPS, filtered=True)
    assert safe.min_h >= -1e-6, safe.min_h
    assert safe.min_dist >= 0.0
    assert float(np.max(safe.viol_resid)) <= 1e-9  # POCS 收尾后约束满足
    print(f"\n[safe] min h {safe.min_h:.2e} m, min net distance {safe.min_dist:.6f} m")


def test_closed_form_projection_is_minimal_modification():
    """闭式解 (7.13)：u* 是 u_des 到可行半空间的最小范数修改（"最小侵入"）。

    - u_des 已安全 → 修改量为零；
    - u_des 违反约束 → u* 恰在边界超平面上（约束取等）；
    - 任意可行对照点到 u_des 的距离都不小于 ‖u* - u_des‖。
    """
    # 已安全情形（教程 07.2 ⑤：u_des 安全时 μ=0，滤波器不干预）：
    # X0 处 h = 1.3 > 0，零控制满足约束。
    u_safe, info = cbf_filter(np.array([0.0, 0.0]), X0, [(CENTER, RADIUS)])
    assert info["viol_before"] == 0.0 and np.allclose(u_safe, 0.0)
    # 违反情形：x 距障碍 0.9 m、以 1.5 m/s 逼近，u_des = [2, 0]
    # （实测 viol_before = 3.9，滤波后 u* = [-7.75, 0]）。
    x = np.array([1.5, 0.0, 1.5, 0.0])
    u_des = np.array([2.0, 0.0])
    u_star, info = cbf_filter(u_des, x, [(CENTER, RADIUS)])
    assert info["viol_before"] > 0.0
    # 约束取等（活跃）：A u* + b = 0。
    assert abs(float((info["A"] @ u_star + info["b"])[0])) < 1e-9
    assert u_star[1] == 0.0  # 修正只在法向（x）方向——最小侵入的几何含义
    # 对照：随机可行的 u 到 u_des 的距离都不小于 u* 的（投影 = 最近可行点）。
    rng = np.random.default_rng(1)
    a, b = info["A"][0], info["b"][0]
    for _ in range(200):
        u_cand = rng.uniform(-12, 12, 2)
        if a @ u_cand + b >= 0.0:  # 可行
            assert np.linalg.norm(u_cand - u_des) >= np.linalg.norm(u_star - u_des) - 1e-9
    # 批量闭式（(L_gh L_ghᵀ + εI) 可逆口径）与逐式闭式一致。
    u_batch = project_halfspaces(u_des, info["A"], info["b"])
    assert np.allclose(u_batch, u_star, atol=1e-6)


def test_filtered_control_reduces_headon_magnitude():
    """迎头场景：滤波只削弱前进分量、从不放大——每步 u_x ≤ u_des_x。

    注意范数语义：进入强刹车段后 ‖u*‖ 可以大于 ‖u_des‖（u_x 变负是
    "刹车"，方向相反）；"最小侵入"的严格口径是测试
    :func:`test_closed_form_projection_is_minimal_modification` 中的
    投影性质，这里断言的是它的前向分量推论。
    """
    safe = simulate_filtered(X0, U_DES, [(CENTER, RADIUS)], DT, N_STEPS, filtered=True)
    assert np.all(safe.u_applied[:, 0] <= U_DES[0] + 1e-9)
    # 且确实做了修改：后期约束活跃，u 被掰向刹车（u_x < 0）。
    assert safe.u_applied[:, 0].min() < -1.0


def test_multi_constraint_projection_iteration_satisfies_all():
    """双障碍同时活跃：批量闭式给初值、POCS 迭代把残余违反压到机器精度。"""
    obstacles = [(np.array([2.4, 0.4]), RADIUS), (np.array([2.4, -0.4]), RADIUS)]
    x = np.array([1.5, 0.0, 1.5, 0.0])
    u_des = np.array([2.0, 0.0])
    u_star, info = cbf_filter(u_des, x, obstacles)
    assert info["viol_before"] > 0.0
    assert info["viol_after"] <= 1e-9  # 多约束同时满足
    slack = info["A"] @ u_star + info["b"]
    # 两条约束同时活跃（夹缝正中）：松弛恰为 POCS/ε 的机器精度量级 ~2e-9。
    assert np.all(np.abs(slack) < 1e-8)
    assert abs(u_star[1]) < 1e-6  # 对称场景：修正纯沿 x
    print(f"\n[multi] u* = {np.round(u_star, 4)}, viol_after = {info['viol_after']:.1e}")


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
