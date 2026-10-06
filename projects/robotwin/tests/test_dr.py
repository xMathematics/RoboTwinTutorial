"""projects/robotwin/dr 的测试 —— 直接运行：``python tests/test_dr.py``。

验证域随机化采样器：同 seed 同参数、none 档 = 名义世界、strong 的每维
区间 ⊇ mild（覆盖条件的单调性）、采样值全部落在物理界内、越界样本被
``clamped`` 截断、非法构造被 ``__post_init__`` 拒绝。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dr import (  # noqa: E402
    NOMINAL_ACTION_SCALE,
    NOMINAL_FRICTION,
    NOMINAL_LINK_LEFT,
    NOMINAL_LINK_RIGHT,
    NOMINAL_OBS_NOISE,
    NOMINAL_RADIUS,
    PHYS_BOUNDS,
    REGIME_BOUNDS,
    WorldParams,
    sample,
)

FIELDS = ("left_link", "right_link", "object_radius",
          "friction", "obs_noise_std", "action_scale")


def _params_equal(a: WorldParams, b: WorldParams) -> bool:
    """逐字段相等（dataclass 未定义逐元组比较，这里显式展开）。"""
    return (
        a.left_link == b.left_link
        and a.right_link == b.right_link
        and a.object_radius == b.object_radius
        and a.friction == b.friction
        and a.obs_noise_std == b.obs_noise_std
        and a.action_scale == b.action_scale
    )


def test_same_seed_same_params():
    """同 seed 的独立 Generator → 逐字段相同的参数序列（mild/strong）。"""
    for regime in ("mild", "strong"):
        for seed in (0, 123):
            ra = sample(regime, np.random.default_rng(seed))
            rb = sample(regime, np.random.default_rng(seed))
            assert _params_equal(ra, rb), regime
    # 同一 Generator 连续采两次必不同（strong 档区间非退化）。
    rng = np.random.default_rng(0)
    assert not _params_equal(sample("strong", rng), sample("strong", rng))


def test_none_regime_is_nominal():
    """none 档 = 名义世界（区间退化 → 任何 rng 都采出同一组名义值）。"""
    for seed in (0, 1, 2026):
        p = sample("none", np.random.default_rng(seed))
        assert p.left_link == NOMINAL_LINK_LEFT
        assert p.right_link == NOMINAL_LINK_RIGHT
        assert p.object_radius == NOMINAL_RADIUS
        assert p.friction == NOMINAL_FRICTION
        assert p.obs_noise_std == NOMINAL_OBS_NOISE
        assert p.action_scale == NOMINAL_ACTION_SCALE
    assert _params_equal(sample("none", np.random.default_rng(0)),
                         WorldParams.nominal())


def test_strong_bounds_contain_mild():
    """strong 的每维采样区间 ⊇ mild 的区间（覆盖条件 (4.2) 的单调性：
    剂量加大 = 随机化范围只扩不缩）。"""
    for key in REGIME_BOUNDS["mild"]:
        lo_m, hi_m = REGIME_BOUNDS["mild"][key]
        lo_s, hi_s = REGIME_BOUNDS["strong"][key]
        assert lo_s <= lo_m and hi_m <= hi_s, key
    # none 是所有区间的公共退化点（= 名义值）。
    for key in REGIME_BOUNDS["none"]:
        lo, hi = REGIME_BOUNDS["none"][key]
        assert lo == hi, key


def test_sampled_params_within_bounds():
    """200 × 3 档采样：每维都落在该档区间与物理界 PHYS_BOUNDS 内。"""
    for regime in ("none", "mild", "strong"):
        bounds = REGIME_BOUNDS[regime]
        for i in range(200):
            p = sample(regime, np.random.default_rng(1000 + i))
            link_lo, link_hi = PHYS_BOUNDS["link"]
            for link in (p.left_link, p.right_link):
                for v in link:
                    assert link_lo <= v <= link_hi
            r_lo, r_hi = bounds["radius_scale"]
            assert r_lo * NOMINAL_RADIUS <= p.object_radius <= r_hi * NOMINAL_RADIUS
            f_lo, f_hi = bounds["friction"]
            assert f_lo <= p.friction <= f_hi
            n_lo, n_hi = bounds["obs_noise_std"]
            assert n_lo <= p.obs_noise_std <= n_hi
            a_lo, a_hi = bounds["action_scale"]
            assert a_lo <= p.action_scale <= a_hi


def test_clamped_clips_out_of_range():
    """越界样本被 clamped 截断回物理界（用 object.__new__ 绕过构造校验，
    模拟"外部输入/优化器产出"的越界参数）。"""
    p = WorldParams.__new__(WorldParams)
    object.__setattr__(p, "left_link", (5.0, -1.0))       # 双双越界
    object.__setattr__(p, "right_link", (0.45, 0.35))
    object.__setattr__(p, "object_radius", 99.0)
    object.__setattr__(p, "friction", 0.0)                # 低于物理下界
    object.__setattr__(p, "obs_noise_std", -0.5)
    object.__setattr__(p, "action_scale", 50.0)
    q = p.clamped()
    link_lo, link_hi = PHYS_BOUNDS["link"]
    assert q.left_link == (link_hi, link_lo)
    assert q.right_link == (0.45, 0.35)                   # 界内不动
    assert q.object_radius == PHYS_BOUNDS["object_radius"][1]
    assert q.friction == PHYS_BOUNDS["friction"][0]
    assert q.obs_noise_std == PHYS_BOUNDS["obs_noise_std"][0]
    assert q.action_scale == PHYS_BOUNDS["action_scale"][1]


def test_invalid_params_rejected():
    """非法构造（负连杆长 / 摩擦越界 / 半径越界）在创建时抛 ValueError。"""
    bad = [
        dict(left_link=(-0.45, 0.35)),                    # 负连杆长
        dict(left_link=(0.02, 0.35)),                     # 低于物理下界
        dict(friction=1.5),                               # 摩擦 > 1
        dict(friction=0.0),                               # 摩擦 = 0（越下界）
        dict(object_radius=1.0),                          # 半径越上界
        dict(obs_noise_std=-0.01),                        # 负噪声
        dict(action_scale=0.0),                           # 零增益
    ]
    for kwargs in bad:
        try:
            WorldParams(
                left_link=kwargs.get("left_link", NOMINAL_LINK_LEFT),
                right_link=kwargs.get("right_link", NOMINAL_LINK_RIGHT),
                object_radius=kwargs.get("object_radius", NOMINAL_RADIUS),
                friction=kwargs.get("friction", NOMINAL_FRICTION),
                obs_noise_std=kwargs.get("obs_noise_std", NOMINAL_OBS_NOISE),
                action_scale=kwargs.get("action_scale", NOMINAL_ACTION_SCALE),
            )
        except ValueError:
            continue
        raise AssertionError(f"非法参数未被拒绝: {kwargs}")
    # 未知档位同样显式报错。
    try:
        sample("extreme", np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("未知档位未被拒绝")


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
