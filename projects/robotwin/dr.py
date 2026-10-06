"""域随机化（Domain Randomization, DR）采样器——迷你基准的"世界参数"发生器。

函数流水线
----------
本模块位于评测流水线的最上游：``benchmark.run`` 在每个回合开始前调用
:func:`sample` 抽出一组 :class:`WorldParams`（臂长/对象半径/摩擦/观测噪声/
动作缩放），交给 ``tasks.py`` 构造回合环境；``policies.py`` 的 robust 策略
则反过来用带噪观测"自标定"这些参数。依赖方向：本模块不依赖库内其他模块
（仅 numpy + 标准库），``tasks / benchmark`` 单向依赖本模块。

与 RoboTwin 2.0 五维 DR 的对照（教程 §4.2.2；论文 §2.2，arXiv:2506.18088）
------------------------------------------------------------------------
RoboTwin 2.0 的五维是"场景杂乱 / 背景纹理 / 光照 / 桌面高度 / 语言指令"，
全部是**视觉-语言维**——纯 NumPy 的 2D 平面世界没有渲染与语言，无法也不应
复刻其内容。本项目做的是**结构对应**：每一维都让"同一脚本策略在标定世界
与随机化世界之间产生同类失配"，从而在 1 分钟内复现教程 §4.2.3 / §6.4 讲的
"DR 剂量 vs 成功率"曲线：

=================  ======================  ====================================
本项目维度         WorldParams 字段        RoboTwin 2.0 对应维（论文 §2.2）
=================  ======================  ====================================
对象尺寸           ``object_radius``       ① 场景杂乱/物体几何（731 物体库）
摩擦               ``friction``            ② 表面物理（纹理+桌高的物理等价物）
臂长               ``left/right_link``     本体参数（5 种本体/体态适配 §4.3）
观测噪声           ``obs_noise_std``       ③ 光照+④ 视觉的感知侧等价物
动作缩放           ``action_scale``        控制接口（部署接口失配）
=================  ======================  ====================================

注：RoboTwin 的第 ⑤ 维"语言指令"在本项目没有对应物（脚本策略不消费语言）；
表中的"本体参数"对应论文的体态（embodiment）轴而非五维 DR 之一，一并列出
是因为它正是教程 §4.3 / §6.3 的主角。

三档剂量（regime）
------------------
``none``  一切取名义值（标定世界，等价 RoboTwin 的 Easy 档）；
``mild``  每维在窄区间内均匀随机（中等剂量）；
``strong`` 每维在宽区间内均匀随机（强剂量，等价 Hard 档）。
区间按"strong 的每维区间 ⊇ mild 的区间"设计（有测试守恒），对应教程
§4.2.1 ⑤ 的覆盖条件 (4.2)：随机化范围扩大 = 训练/测试分布差距被包进
随机化区间。同一 ``regime`` + 同一 ``rng`` 状态给出同一参数序列（确定性）。

输入/输出：输入档位名与 ``numpy`` Generator；输出一个 :class:`WorldParams`
（单位 m / 无量纲）。本模块不产生任何回合动态——那是 ``tasks.py`` 的事。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "REGIMES",
    "WorldParams",
    "NOMINAL_LINK_LEFT",
    "NOMINAL_LINK_RIGHT",
    "NOMINAL_RADIUS",
    "NOMINAL_FRICTION",
    "NOMINAL_OBS_NOISE",
    "NOMINAL_ACTION_SCALE",
    "PHYS_BOUNDS",
    "sample",
    "regime_bounds",
]

#: 支持的剂量档位（顺序即 benchmark 遍历顺序，索引参与种子派生）。
REGIMES: tuple[str, ...] = ("none", "mild", "strong")

# ---------------------------------------------------------------- 名义世界 --
#: 名义连杆长 (L1, L2)，单位 m（左臂 = 右臂；"标定即完美"的 CAD 值）。
NOMINAL_LINK_LEFT: tuple[float, float] = (0.45, 0.35)
NOMINAL_LINK_RIGHT: tuple[float, float] = (0.45, 0.35)
#: 名义对象半径（圆盘/方块外接半径），单位 m。
NOMINAL_RADIUS: float = 0.05
#: 名义摩擦系数（接触时对象位移 / 末端位移），无量纲，(0, 1]。
NOMINAL_FRICTION: float = 1.0
#: 名义观测噪声标准差，单位 m（none 档 = 完美感知）。
NOMINAL_OBS_NOISE: float = 0.0
#: 名义动作缩放（控制接口增益），无量纲，1 = 无失配。
NOMINAL_ACTION_SCALE: float = 1.0

#: 物理合理边界（§4.2.3 剂量准则之二："物理合理范围"）。
#: 任何 WorldParams（无论来自 sample 还是手工构造）超出这些界即非法：
#: 构造时校验抛 ValueError，:meth:`WorldParams.clamped` 截断回界内。
PHYS_BOUNDS: dict[str, tuple[float, float]] = {
    "link": (0.05, 1.0),        # 每节连杆长，m
    "object_radius": (0.01, 0.20),  # 对象半径，m
    "friction": (0.05, 1.0),    # 摩擦系数，无量纲
    "obs_noise_std": (0.0, 0.10),   # 观测噪声标准差，m
    "action_scale": (0.10, 2.0),    # 动作缩放，无量纲
}

#: 三档剂量的每维采样区间（缩放系数区间或绝对区间）。
#: 设计约束（有测试守恒）：strong 的每维区间 ⊇ mild 的区间；none 全为
#: 退化区间 = 名义值。arm/radius 用"相对名义值的缩放系数"，其余为绝对值。
REGIME_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "none": {
        "arm_scale": (1.0, 1.0),
        "radius_scale": (1.0, 1.0),
        "friction": (1.0, 1.0),
        "obs_noise_std": (0.0, 0.0),
        "action_scale": (1.0, 1.0),
    },
    "mild": {
        "arm_scale": (0.95, 1.05),      # 臂长 ±5%：标定轻微失准
        "radius_scale": (0.80, 1.20),   # 对象尺寸 ±20%
        "friction": (0.85, 1.0),        # 桌面略滑
        "obs_noise_std": (0.0, 0.006),  # 轻微感知噪声（m）
        "action_scale": (0.95, 1.05),   # 接口增益 ±5%
    },
    "strong": {
        "arm_scale": (0.85, 1.15),      # 臂长 ±15%：标定严重失准
        "radius_scale": (0.60, 1.40),   # 对象尺寸 ±40%
        "friction": (0.70, 1.0),        # 桌面明显变滑
        "obs_noise_std": (0.0, 0.015),  # 显著感知噪声（m）
        "action_scale": (0.85, 1.15),   # 接口增益 ±15%
    },
}


@dataclass(frozen=True)
class WorldParams:
    """一个回合的"世界参数"——DR 采样结果的载体（不可变值对象）。

    在 2D 平面世界里，这组参数完整决定回合物理：双臂运动学（连杆长）、
    对象几何（半径）、接触动力学（摩擦）、感知（观测噪声）与控制接口
    （动作缩放）。``tasks.py`` 的环境用真值演化世界，再按 ``obs_noise_std``
    生成观测；``policies.py`` 的策略只从观测推断这些参数。

    Attributes:
        left_link: (L1, L2) 左臂两节连杆长，单位 m，取值见 ``PHYS_BOUNDS["link"]``。
        right_link: (L1, L2) 右臂两节连杆长，单位 m。
        object_radius: 对象（圆盘）半径，单位 m。
        friction: 摩擦系数（接触时对象位移 = friction × 末端位移），
            无量纲，(0, 1]。
        obs_noise_std: 末端/对象位置观测的高斯噪声标准差，单位 m，≥ 0。
        action_scale: 动作缩放——关节伺服对"目标−当前"增量乘的增益，
            无量纲，> 0；≠ 1 即控制接口失配（教程 §4.2 的"接口"维）。

    Raises:
        ValueError: 任一字段超出 :data:`PHYS_BOUNDS` 物理界（构造即校验，
            早失败优于静默传播非法世界）。
    """

    left_link: tuple[float, float]
    right_link: tuple[float, float]
    object_radius: float
    friction: float
    obs_noise_std: float
    action_scale: float

    def __post_init__(self) -> None:
        # 构造即校验（依据：§4.2.3 "物理合理范围"剂量准则；非法世界应
        # 在创建处报错，而不是在回合中途产生不可复现的怪现象）。
        lo, hi = PHYS_BOUNDS["link"]
        for name in ("left_link", "right_link"):
            link = tuple(float(v) for v in getattr(self, name))
            if len(link) != 2 or not (lo <= link[0] <= hi and lo <= link[1] <= hi):
                raise ValueError(
                    f"{name} 必须是两节连杆长且每节在 [{lo}, {hi}] m 内，"
                    f"当前为 {getattr(self, name)!r}"
                )
        r_lo, r_hi = PHYS_BOUNDS["object_radius"]
        if not r_lo <= self.object_radius <= r_hi:
            raise ValueError(f"object_radius 须在 [{r_lo}, {r_hi}] m 内，当前 {self.object_radius}")
        f_lo, f_hi = PHYS_BOUNDS["friction"]
        if not f_lo <= self.friction <= f_hi:
            raise ValueError(f"friction 须在 [{f_lo}, {f_hi}] 内，当前 {self.friction}")
        n_lo, n_hi = PHYS_BOUNDS["obs_noise_std"]
        if not n_lo <= self.obs_noise_std <= n_hi:
            raise ValueError(f"obs_noise_std 须在 [{n_lo}, {n_hi}] m 内，当前 {self.obs_noise_std}")
        a_lo, a_hi = PHYS_BOUNDS["action_scale"]
        if not a_lo <= self.action_scale <= a_hi:
            raise ValueError(f"action_scale 须在 [{a_lo}, {a_hi}] 内，当前 {self.action_scale}")

    @classmethod
    def nominal(cls) -> "WorldParams":
        """名义世界（CAD 标定值，即 none 档），等价 RoboTwin 的 Easy 档环境。"""
        return cls(
            left_link=NOMINAL_LINK_LEFT,
            right_link=NOMINAL_LINK_RIGHT,
            object_radius=NOMINAL_RADIUS,
            friction=NOMINAL_FRICTION,
            obs_noise_std=NOMINAL_OBS_NOISE,
            action_scale=NOMINAL_ACTION_SCALE,
        )

    def clamped(self) -> "WorldParams":
        """返回把每个字段截断回物理界 :data:`PHYS_BOUNDS` 后的副本。

        用途：外部手工构造/优化器产出的参数可能越界（如消融脚本把摩擦推
        到 0），:func:`sample` 的均匀采样不会越界但公开接口可能——截断
        （而非报错）让"越界样本"可被稳妥地消费；需要硬失败时直接构造
        （``__post_init__`` 会抛 :class:`ValueError`）。
        """
        lo, hi = PHYS_BOUNDS["link"]
        clamp = lambda v: float(min(max(v, lo), hi))  # noqa: E731 —— 每节连杆共用界
        r_lo, r_hi = PHYS_BOUNDS["object_radius"]
        f_lo, f_hi = PHYS_BOUNDS["friction"]
        n_lo, n_hi = PHYS_BOUNDS["obs_noise_std"]
        a_lo, a_hi = PHYS_BOUNDS["action_scale"]
        return replace(
            self,
            left_link=(clamp(self.left_link[0]), clamp(self.left_link[1])),
            right_link=(clamp(self.right_link[0]), clamp(self.right_link[1])),
            object_radius=float(min(max(self.object_radius, r_lo), r_hi)),
            friction=float(min(max(self.friction, f_lo), f_hi)),
            obs_noise_std=float(min(max(self.obs_noise_std, n_lo), n_hi)),
            action_scale=float(min(max(self.action_scale, a_lo), a_hi)),
        )


def regime_bounds(regime: str) -> dict[str, tuple[float, float]]:
    """返回某剂量档位的每维采样区间（只读视图的浅拷贝）。

    Args:
        regime: 剂量档位名，取值 ``"none" / "mild" / "strong"``。

    Returns:
        dict，键为 ``arm_scale / radius_scale / friction / obs_noise_std /
        action_scale``，值为闭区间 ``(下界, 上界)``；arm/radius 两键是
        相对名义值的缩放系数区间，其余为绝对值区间。

    Raises:
        ValueError: 档位名不认识时抛出（列出可用档位）。
    """
    if regime not in REGIME_BOUNDS:
        raise ValueError(f"未知 DR 档位 {regime!r}；可用：{sorted(REGIME_BOUNDS)}")
    return dict(REGIME_BOUNDS[regime])


def sample(regime: str, rng: np.random.Generator) -> WorldParams:
    """按剂量档位采一组世界参数（确定性：同 rng 状态同结果）。

    每维独立均匀采样（依据：RoboTwin 2.0 附录 C 对桌高等维采用合理区间
    均匀随机的口径；教程 §4.2.2 "在合理范围内随机采样"）。``none`` 档的
    区间全部退化，采样结果恒等于名义世界——因此"none"与"未开 DR"严格
    等价，剂量曲线以它为 100% 基线。

    Args:
        regime: 剂量档位名（``"none" / "mild" / "strong"``）。
        rng: ``numpy`` 随机数生成器；调用方保证每回合用独立派生的 rng
            （见 ``benchmark`` 的种子规则），使整条评测流水线确定可复现。

    Returns:
        :class:`WorldParams`：一组物理合法的世界参数。

    Raises:
        ValueError: 档位名不认识时抛出。
    """
    bounds = regime_bounds(regime)
    # 左右臂各自独立采一个缩放系数（本体维的两个自由度）。
    s_left = float(rng.uniform(*bounds["arm_scale"]))
    s_right = float(rng.uniform(*bounds["arm_scale"]))
    s_radius = float(rng.uniform(*bounds["radius_scale"]))
    friction = float(rng.uniform(*bounds["friction"]))
    noise = float(rng.uniform(*bounds["obs_noise_std"]))
    action = float(rng.uniform(*bounds["action_scale"]))
    return WorldParams(
        left_link=(
            NOMINAL_LINK_LEFT[0] * s_left,
            NOMINAL_LINK_LEFT[1] * s_left,
        ),
        right_link=(
            NOMINAL_LINK_RIGHT[0] * s_right,
            NOMINAL_LINK_RIGHT[1] * s_right,
        ),
        object_radius=NOMINAL_RADIUS * s_radius,
        friction=friction,
        obs_noise_std=noise,
        action_scale=action,
    )
