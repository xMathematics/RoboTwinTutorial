"""projects/robotwin/arm 的测试 —— 直接运行：``python tests/test_arm.py``。

验证平面 2 连杆双臂运动学（教程第 02 章 §2.1 的 FK/IK/雅可比）：
FK 与手算一致、解析 IK 回代 FK 闭合到机器精度、解析雅可比与中心差分
一致、肘向双解互异且同精度、不可达目标与限位检查的行为，以及
``BimanualArm2D`` 双臂状态的分块一致性。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm import (  # noqa: E402
    BASES,
    BimanualArm2D,
    LEFT,
    Q1_LIMITS,
    Q2_LIMITS,
    RIGHT,
    ik_both,
    link_fk,
    link_ik,
    link_jacobian,
    numerical_jacobian,
)
from dr import NOMINAL_LINK_LEFT  # noqa: E402

LINKS = tuple(NOMINAL_LINK_LEFT)


def test_fk_matches_hand_computation():
    """FK 与手算一致：q=(0.9, -1.8)、基座原点时
    x = 0.45·cos(0.9) + 0.35·cos(-0.9)，y = 0.45·sin(0.9) + 0.35·sin(-0.9)
    （依据：平面开链逐节旋转叠加的解析式，教程 (2.1) 的平面版）。"""
    q = np.array([0.9, -1.8])
    expect_x = 0.45 * np.cos(0.9) + 0.35 * np.cos(-0.9)
    expect_y = 0.45 * np.sin(0.9) + 0.35 * np.sin(-0.9)
    xy = link_fk(LINKS, q, base=(0.0, 0.0))
    assert abs(xy[0] - expect_x) < 1e-12, xy
    assert abs(xy[1] - expect_y) < 1e-12, xy
    # 非零基座 = 平移（双臂布局的肩部偏移）。
    xy2 = link_fk(LINKS, q, base=BASES[LEFT])
    assert np.allclose(xy2, xy + np.asarray(BASES[LEFT]), atol=1e-15)


def test_ik_roundtrip_error_below_1e_8():
    """IK 解回代 FK 的误差 < 1e-8：可达网格上两个肘向各解一支。

    门限理由：解析几何法（余弦定理 + atan2 回代）无迭代，误差只来自
    浮点舍入（~1e-16 量级）；1e-8 已留 8 个数量级裕量。
    """
    worst = 0.0
    for tx in np.linspace(0.2, 0.75, 6):
        for ty in np.linspace(-0.6, 0.6, 7):
            for elbow in (+1.0, -1.0):
                q = link_ik(LINKS, np.array([tx, ty]), elbow=elbow, joint_limits=False)
                if q is None:
                    continue  # 该点该肘向不可达，跳过（不是本测试的对象）
                back = link_fk(LINKS, q)
                err = float(np.linalg.norm(back - np.array([tx, ty])))
                worst = max(worst, err)
    assert worst < 1e-8, worst
    print(f"\n[ik-roundtrip] worst |FK(IK(x)) - x| = {worst:.2e} (< 1e-8)")


def test_analytic_jacobian_matches_central_difference():
    """解析雅可比与中心差分一致（< 1e-6）。

    门限理由：中心差分截断误差 O(eps^2)，eps=1e-6 时 ~1e-12，加上舍入
    实测 ~1e-9；1e-6 是保守门限。"""
    worst = 0.0
    for q in (np.array([0.9, -1.8]), np.array([0.3, -0.7]), np.array([1.6, -2.5])):
        j_ana = link_jacobian(LINKS, q)
        j_num = numerical_jacobian(lambda qq: link_fk(LINKS, qq), q)
        worst = max(worst, float(np.max(np.abs(j_ana - j_num))))
    assert worst < 1e-6, worst
    print(f"\n[jacobian] max |J_analytic - J_numeric| = {worst:.2e} (< 1e-6)")


def test_two_elbow_solutions_differ():
    """肘向双解：q2 符号相反、q1 不同，但回代 FK 到同一点（< 1e-9）。"""
    target = np.array([0.5, 0.2])
    plus, minus = ik_both(LINKS, target)
    assert plus is not None and minus is not None
    assert np.sign(plus[1]) > 0 and np.sign(minus[1]) < 0  # q2 符号相反
    assert abs(plus[1] + minus[1]) < 1e-12  # arccos 的偶对称性
    assert abs(plus[0] - minus[0]) > 1e-3  # q1 确实不同（不是同一解）
    for sol in (plus, minus):
        back = link_fk(LINKS, sol)
        assert np.linalg.norm(back - target) < 1e-9


def test_ik_unreachable_returns_none():
    """目标超出可达工作空间（r > L1+L2）→ 双解皆 None（不做静默外推）。"""
    far = np.array([LINKS[0] + LINKS[1] + 0.05, 0.0])
    plus, minus = ik_both(LINKS, far)
    assert plus is None and minus is None
    assert link_ik(LINKS, far) is None


def test_ik_respects_joint_limits():
    """限位检查：elbow=+1 一支的 q2 > 0 违反 Q2_LIMITS（上限为负）→
    默认入口返回 None；joint_limits=False 时保留原始解。"""
    target = np.array([0.5, 0.2])
    raw_plus = link_ik(LINKS, target, elbow=+1.0, joint_limits=False)
    assert raw_plus is not None and raw_plus[1] > Q2_LIMITS[1]
    assert link_ik(LINKS, target, elbow=+1.0) is None  # 限位内不可行
    got = link_ik(LINKS, target, elbow=-1.0)  # 工作肘向应在限位内
    assert got is not None
    assert Q1_LIMITS[0] <= got[0] <= Q1_LIMITS[1]
    assert Q2_LIMITS[0] <= got[1] <= Q2_LIMITS[1]


def test_bimanual_state_and_per_arm_consistency():
    """双臂状态 (4,) 分块 = [左(2), 右(2)]：end_effectors 的每行与单臂
    FK（含各自基座）一致，ik 方法与模块级 link_ik 一致。"""
    rng = np.random.default_rng(7)
    q = rng.uniform([-1.0, -2.8, -1.0, -2.8], [2.0, -0.3, 2.0, -0.3])
    arm = BimanualArm2D(q0=q)
    assert arm.get_state().shape == (4,)
    assert np.array_equal(arm.get_state(), q)
    ee = arm.end_effectors()
    assert ee.shape == (2, 2)
    for side in (LEFT, RIGHT):
        expect = link_fk(arm.links[side], q[2 * side: 2 * side + 2], BASES[side])
        assert np.allclose(ee[side], expect, atol=1e-12)
        target = np.array([0.45, 0.25 * (1 if side == LEFT else -1)])
        assert np.allclose(
            arm.ik(side, target), link_ik(arm.links[side], target, BASES[side])
        )
    arm.set_state(np.zeros(4))
    assert np.array_equal(arm.get_state(), np.zeros(4))


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
