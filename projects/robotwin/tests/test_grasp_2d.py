"""projects/robotwin/grasp_2d 的测试 —— 直接运行：``python tests/test_grasp_2d.py``。

验证 DexGraspNet 教学代理（2D 平行夹爪力闭合 + Ferrari–Canny L1）的行为
主张（对应精读 DexGraspNet §4.2–4.6 与教程第 09 章 §9.2 / 3D 重建第 10 章
§10.3 式 (10.4)(10.5)）：
- 正方形对径抓取力闭合成立且质量 > 0；相邻边抓取不成立；
- 偏移对径抓取的力闭合边界与手算一致（Nguyen 连线在锥内：arctan(c/2a) ≤ arctan μ）；
- 摩擦系数 μ = 0 时任何抓取都不满足"内点"判据（无摩擦的边界情形）；
- 圆盘最优候选回到对径（且质量排序正确）；
- L1 与支撑函数方向采样（独立口径）交叉验证一致；
- 质量随摩擦系数单调不减（锥变宽 → 凸包变大）；
- 与本项目 tasks.py pick_place 的联动 sanity（最优抓取点满足吸附条件）；
- 物体构造校验（非凸/CW/非正半径被拒绝）。

全部用例确定性（无随机数；候选为固定网格），门限旁注明实测值与裕量。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grasp_2d as g  # noqa: E402

#: 测试用正方形（边长 1 m，中心在原点，CCW 顶点）。
SQUARE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


def _square_antipodal() -> g.Grasp2D:
    """正方形上下边中点的对径抓取（接触点 (0, ∓0.5)，内法向相对）。"""
    poly = g.ConvexPolygon(SQUARE)
    p1, n1 = poly.contact(0, 0.5)  # 下边中点，内法向 (0, 1)
    p2, n2 = poly.contact(2, 0.5)  # 上边中点，内法向 (0, −1)
    return g.Grasp2D(np.stack([p1, p2]), np.stack([n1, n2]))


def _square_offset(c: float) -> g.Grasp2D:
    """偏移对径抓取：接触点 (−0.5, 0) 与 (0.5, c)（右边上偏移 c 的点），
    内法向 (±1, 0)。力闭合边界（手算，Nguyen 连线在锥内）：
    连线倾角 arctan(c/1) ≤ arctan μ ⟺ c ≤ μ（接触距 2a = 1）。"""
    pts = np.array([[-0.5, 0.0], [0.5, c]])
    nrm = np.array([[1.0, 0.0], [-1.0, 0.0]])
    return g.Grasp2D(pts, nrm)


def test_square_antipodal_force_closure_positive_quality():
    """正方形对径（上下边中点）抓取：力闭合成立、L1 > 0（实测 0.167，
    门限 0.05 留 ~3 倍裕量）；张开宽度 = 边长 1、拟合轴沿 y。"""
    grasp = _square_antipodal()
    assert g.is_force_closure(grasp, mu=0.4)
    q = g.grasp_quality(grasp, mu=0.4)
    assert q > 0.05, q
    assert abs(grasp.width - 1.0) < 1e-12
    assert abs(abs(float(np.dot(grasp.axis, np.array([0.0, 1.0])))) - 1.0) < 1e-12
    print(f"\n[square-antipodal] FC=True, L1={q:.5f}（μ=0.4，m=8 射线）")


def test_square_offset_antipodal_friction_boundary():
    """偏移对径的力闭合边界与手算一致（本文件的定量锚点）：μ=0.4 时
    c=0.25（< 0.4）成立、c=0.45（> 0.4）不成立；μ 加大到 0.5 后 c=0.45
    重新成立——边界 c = 2aμ 随摩擦移动（实测 L1 = 0.060 / 0 / 0.018）。"""
    assert g.is_force_closure(_square_offset(0.25), mu=0.4)
    q_ok = g.grasp_quality(_square_offset(0.25), mu=0.4)
    assert q_ok > 0.01, q_ok
    assert not g.is_force_closure(_square_offset(0.45), mu=0.4)
    assert g.grasp_quality(_square_offset(0.45), mu=0.4) == 0.0
    assert g.is_force_closure(_square_offset(0.45), mu=0.5)
    print(f"\n[offset-boundary] c=0.25@μ0.4 L1={q_ok:.5f}；c=0.45@μ0.4=0；c=0.45@μ0.5=True")


def test_square_adjacent_edge_grasp_not_force_closure():
    """明显偏离对径的抓取（右边中点 + 上边中点，法向夹角 90°）不力闭合
    （Nguyen 必要条件：连线不在锥内），质量按协议返回 0。"""
    poly = g.ConvexPolygon(SQUARE)
    p1, n1 = poly.contact(1, 0.5)  # 右边中点，内法向 (−1, 0)
    p2, n2 = poly.contact(2, 0.5)  # 上边中点，内法向 (0, −1)
    adj = g.Grasp2D(np.stack([p1, p2]), np.stack([n1, n2]))
    assert not g.is_force_closure(adj, mu=0.4)
    assert g.grasp_quality(adj, mu=0.4) == 0.0
    # 独立口径交叉印证：支撑函数最小值明显为负（实测 −0.387，原点在凸包外）。
    signed = g.l1_by_support_sampling(g.grasp_wrenches(adj, 0.4), n_dirs=4000)
    assert signed < -0.1, signed
    print(f"\n[adjacent] FC=False, L1=0, 支撑函数口径 = {signed:.3f}（< 0，包外）")


def test_frictionless_contacts_fail_interior_criterion():
    """μ = 0（无摩擦）的边界情形：锥退化为单射线 → 凸包至多是一条线段，
    原点永不成为 3D 凸包**内点** → 力闭合判 False（"无摩擦对径抓取只是
    临界平衡，不能力闭合"的经典事实；μ > 0 立即恢复）。"""
    grasp = _square_antipodal()
    assert not g.is_force_closure(grasp, mu=0.0)
    assert g.grasp_quality(grasp, mu=0.0) == 0.0
    disk = g.Disk(np.zeros(2), 0.3)
    p1, n1 = disk.contact(0.0)
    p2, n2 = disk.contact(np.pi)
    anti = g.Grasp2D(np.stack([p1, p2]), np.stack([n1, n2]))
    assert not g.is_force_closure(anti, mu=0.0)
    assert g.is_force_closure(anti, mu=0.4)
    print("\n[frictionless] μ=0 → 对径抓取判 False（内点判据）；μ=0.4 → True")


def test_disk_best_candidate_is_antipodal():
    """圆盘（R=0.3，12 等分角网格 66 候选）按质量排序：最优 = 精确对径
    （实测 L1 = 0.107，6 个对径对并列且按生成序取第一个）；对径质量严格
    大于最近偏对径（30° 偏移，实测 0.033）——"最优抓取回到对径"。"""
    disk = g.Disk(np.zeros(2), 0.3)
    cands = g.disk_grasp_candidates(disk, n_angles=12)
    assert len(cands) == 66
    ranked = g.rank_candidates(cands, mu=0.4)
    best_q, best = ranked[0]
    angle = lambda p: float(np.arctan2(p[1], p[0]))
    diff = abs(angle(best.points[1]) - angle(best.points[0]))
    diff = min(diff, 2.0 * np.pi - diff)
    assert abs(diff - np.pi) < 1e-9, diff  # 最优候选精确对径
    assert best_q > 0.05, best_q
    non_anti = [q for q, gr in ranked
                if q > 0 and abs(min(abs(angle(gr.points[1]) - angle(gr.points[0])),
                                     2 * np.pi - abs(angle(gr.points[1]) - angle(gr.points[0]))) - np.pi) > 1e-9]
    assert non_anti and best_q > max(non_anti) + 0.02, (best_q, max(non_anti))
    # 并列的对径对质量逐位相等（圆对称性），best_grasp 确定取生成序第一个。
    anti_pairs = [gr for q, gr in ranked if q > 0 and
                  abs(min(abs(angle(gr.points[1]) - angle(gr.points[0])),
                          2 * np.pi - abs(angle(gr.points[1]) - angle(gr.points[0]))) - np.pi) < 1e-9]
    assert len(anti_pairs) == 6
    assert all(abs(g.grasp_quality(gr, 0.4) - best_q) < 1e-12 for gr in anti_pairs)
    assert g.best_grasp(cands, mu=0.4) is ranked[0][1]
    print(f"\n[disk-best] 66 候选，最优 L1={best_q:.5f} 为对径；偏对径最高 {max(non_anti):.5f}")


def test_l1_cross_validated_by_support_sampling():
    """L1 交叉验证：wrench_hull_analysis（支撑平面枚举）与
    l1_by_support_sampling（支撑函数方向采样，独立实现）在力闭合抓取上
    一致（gap ≤ 网格间距 × 最大 wrench 范数；实测 ≤ 0.005，门限 0.05）。"""
    disk = g.Disk(np.zeros(2), 0.3)
    checked = 0
    for ang_deg in (0.0, 30.0, 60.0, 90.0):
        a = np.radians(ang_deg)
        p = np.array([[0.3 * np.cos(a), 0.3 * np.sin(a)],
                      [-0.3 * np.cos(a), -0.3 * np.sin(a)]])
        gr = g.Grasp2D(p, -p / 0.3)
        q = g.grasp_quality(gr, 0.4)
        assert q > 0.0
        sampled = g.l1_by_support_sampling(g.grasp_wrenches(gr, 0.4), n_dirs=4000)
        assert 0.0 <= sampled - q < 0.05, (ang_deg, q, sampled)  # 栅格 min ≥ 真值
        checked += 1
    poly = g.ConvexPolygon(SQUARE)
    q = g.grasp_quality(_square_antipodal(), 0.4)
    sampled = g.l1_by_support_sampling(g.grasp_wrenches(_square_antipodal(), 0.4), n_dirs=4000)
    assert 0.0 <= sampled - q < 0.05, (q, sampled)
    print(f"\n[cross-val] {checked + 1} 个力闭合抓取两种口径一致（最大 gap < 0.05）")


def test_quality_monotone_in_friction():
    """质量随摩擦系数单调不减（锥变宽 → wrench 凸包变大 → 内切半径变大；
    离散射线数固定 m=8。实测对径 0.044→0.088→0.129→0.167）。"""
    grasp = _square_antipodal()
    qs = [g.grasp_quality(grasp, mu) for mu in (0.1, 0.2, 0.3, 0.4)]
    assert all(b >= a for a, b in zip(qs, qs[1:])), qs
    offset = _square_offset(0.25)
    qo = [g.grasp_quality(offset, mu) for mu in (0.2, 0.3, 0.4)]
    assert qo[0] == 0.0 and qo[1] > 0.0 and qo[2] > qo[1], qo  # 越过边界后仍单调
    print(f"\n[monotone] 对径 μ=0.1..0.4 → {[round(q, 4) for q in qs]}；偏移 c=0.25 → {[round(q, 4) for q in qo]}")


def test_pick_place_grasp_points_sanity():
    """与 tasks.py pick_place 的联动 sanity：把对象建为半径 = 名义
    NOMINAL_RADIUS 的圆盘，最优抓取的两接触点都满足吸附判定
    ‖ee − 圆心‖ ≤ R + GRASP_PAD（即最优抓取点可直接作为放置交互点），
    且接触对中点 = 圆心（对径抓取捏住直径）。"""
    import dr
    import tasks as tk
    radius = dr.NOMINAL_RADIUS
    center = np.array([0.3, 0.2])
    disk = g.Disk(center, radius)
    best = g.best_grasp(g.disk_grasp_candidates(disk, 12), mu=0.4)
    for p in best.points:
        assert float(np.linalg.norm(p - center)) <= radius + tk.GRASP_PAD + 1e-12, p
    assert float(np.linalg.norm(best.points.mean(axis=0) - center)) < 1e-9
    assert abs(best.width - 2.0 * radius) < 1e-9
    print(f"\n[pick-place] 最优抓取点距圆心 ≤ R+GRASP_PAD = {radius + tk.GRASP_PAD}，中点=圆心")


def test_polygon_and_disk_validation_rejects_bad_input():
    """物体/抓取构造校验（早失败约定）：CW 顶点、凹多边形、非正半径、
    非单位法向都被 ValueError 拒绝。"""
    bad_inputs = [
        lambda: g.ConvexPolygon(SQUARE[::-1].copy()),  # CW 序
        lambda: g.ConvexPolygon(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
                                          [0.4, 0.2], [0.0, 0.5]])),  # 凹
        lambda: g.Disk(np.zeros(2), 0.0),
        lambda: g.Grasp2D(np.zeros((2, 2)), np.array([[1.0, 0.0], [0.0, 2.0]])),  # 法向非单位
    ]
    for call in bad_inputs:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"非法输入未被拒绝: {call}")
    print("\n[validation] CW/凹/零半径/非单位法向均被 ValueError 拒绝")


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
