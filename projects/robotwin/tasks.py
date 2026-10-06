"""三种双臂任务回合环境（reach / push / pick_place）——迷你基准的"世界"。

函数流水线
----------
本模块在评测流水线的中游：``benchmark.run`` 每回合先经 ``dr.sample`` 抽
世界参数，再用 :func:`make_task` 构造本模块的环境实例；回合循环为
``reset(seed, split) → obs``、反复 ``step(action) → obs, reward, done, info``
（``info["success"]`` 是评测协议的唯一结果变量，交给 ``metrics.py`` 聚合）。
``policies.py`` 的脚本策略消费 ``obs``、产出 ``action``。依赖方向：
``tasks → arm``（FK/限位）与 ``tasks → dr``（WorldParams 取真值参数）。

回合接口（统一约定）
--------------------
- ``reset(seed, split) -> obs``：``seed`` 决定场景采样与全部观测噪声
  （确定性：同 seed 逐位一致）；``split ∈ {"seen", "unseen"}`` 选目标
  位姿的采样范围（评测协议的泛化分层，教程 §5.6 / §6.5 的 seen/unseen
  口径——seen 在标定工作空间内部，unseen 在外围一圈）。
- ``step(action) -> (obs, reward, done, info)``：动作 ``action = (5,)``
  ``= [qL1, qL2, qR1, qR2, grip]``，前 4 项是双臂关节目标（rad，低层为
  限速关节伺服），第 5 项是夹爪指令（< 0.5 开、≥ 0.5 闭；reach/push 忽略）。

任务物理（2D 化简，对应教程 §2.2 双臂任务的三类基元）
----------------------------------------------------
- ``reach``：双臂末端各自到达目标点，成功 = 两臂距离都 < :data:`REACH_TOL`。
- ``push``：末端把圆盘推到目标点。接触判定 ``‖ee − 圆心‖ ≤ R + PUSH_PAD``；
  接触时圆盘沿末端位移方向移动 ``friction × |ee 位移|``（简单摩擦滑动模型，
  对应论文 Table 5 任务族 push 类的 2D 化简）。
- ``pick_place``：夹爪状态机 ``open → (对准中心闭合) attached → (搬运)
  → (松开) released``。只有**闭合的**夹爪且 ``‖ee − 圆心‖ ≤ R + GRASP_PAD``
  才吸附；吸附后对象随持握臂刚体平移（保存吸附偏移）；松开后对象停留。
  成功 = **松开后**对象距目标 < :data:`PLACE_TOL`（必须完成"放置"动作，
  拎着不算）。开爪接触不推动对象——对应 3D 里"从上方接近不会碰歪物体"
  的 2D 化简，也让状态机语义可被测试精确检验。

观测（obs，dict of numpy 数组；噪声只加在"感知"字段上）
----------------------------------------------------
=================  ===========  =============================================
键                 形状         含义 / 噪声
=================  ===========  =============================================
``q``              (4,)         双臂关节角（本体感知，无噪声），rad
``ee``             (2, 2)       双臂末端位置，行 = [左, 右]；+ 高斯噪声，m
``object``         (2,)         对象位置（reach 任务为零占位）；+ 噪声，m
``has_object``     () bool      场景中是否有对象
``radius``         () float64   对象半径读数；+ 噪声，m
``goal``           (2, 2)/(2,)  目标位姿：reach 为双臂目标点对，push/pick 为
                                对象目标点；**无噪声**（目标=任务指令）
``holding``        () bool      pick_place：对象是否已被吸附
=================  ===========  =============================================
"""
from __future__ import annotations

import numpy as np

from arm import BimanualArm2D, LEFT, Q1_LIMITS, Q2_LIMITS, RIGHT
from dr import WorldParams

__all__ = [
    "TASKS",
    "REACH_TOL",
    "PUSH_TOL",
    "PLACE_TOL",
    "GRASP_PAD",
    "PUSH_PAD",
    "MAX_STEPS",
    "JOINT_SPEED",
    "HOME_Q",
    "SCENE_RANGES",
    "BaseBimanualEnv",
    "ReachTask",
    "PushTask",
    "PickPlaceTask",
    "make_task",
]

#: 任务名注册表（顺序即 benchmark 遍历顺序，索引参与种子派生）。
TASKS: tuple[str, ...] = ("reach", "push", "pick_place")

#: reach 成功容差：双臂末端到各自目标点的最大距离，单位 m。
REACH_TOL: float = 0.035
#: push 成功容差：圆盘圆心到目标点的距离，单位 m。
PUSH_TOL: float = 0.055
#: pick_place 成功容差：松开后对象圆心到目标点的距离，单位 m。
PLACE_TOL: float = 0.05
#: 抓取接触判定余量：吸附条件 ‖ee − 圆心‖ ≤ R + GRASP_PAD，单位 m。
GRASP_PAD: float = 0.015
#: push 接触判定余量：‖ee − 圆心‖ ≤ R + PUSH_PAD，单位 m。
PUSH_PAD: float = 0.012
#: 回合最大步数（超时未成功 = 失败，done=True）。
MAX_STEPS: int = 64
#: 关节伺服每步最大角位移，单位 rad/步（限速 → 末端线速度 ≈ 0.12 m/步）。
JOINT_SPEED: float = 0.15
#: 回合初始（home）双臂关节角 [qL1, qL2, qR1, qR2]，单位 rad。
HOME_Q: np.ndarray = np.array([0.9, -1.8, 0.9, -1.8])

#: 场景采样范围（单位 m，桌面系）。seen = 标定工作空间内部；unseen = 外围
#: 一圈（仍保证在任何 DR 档的最短臂长下物理可达——排除"任务本身不可能"
#: 的样本，使剂量曲线只反映策略失配而非不可达）。
#: y 记 |y|，符号（左/右侧）由每回合 rng 独立抽取：左臂负责 y>0 一侧、
#: 右臂负责 y<0 一侧（教程 §2.2 双臂分工的 2D 化简）。
SCENE_RANGES: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "reach": {
        "seen": {"x": (0.34, 0.46), "y_abs": (0.14, 0.30)},
        "unseen": {"x": (0.55, 0.63), "y_abs": (0.26, 0.36)},
    },
    "push": {
        "start": {"x": (0.26, 0.34), "y_abs": (0.12, 0.22)},
        "seen": {"x": (0.44, 0.52), "y_abs": (0.14, 0.26)},
        "unseen": {"x": (0.56, 0.62), "y_abs": (0.22, 0.30)},
    },
    "pick_place": {
        "start": {"x": (0.28, 0.34), "y_abs": (0.12, 0.20)},
        "seen": {"x": (0.46, 0.54), "y_abs": (0.16, 0.28)},
        "unseen": {"x": (0.56, 0.62), "y_abs": (0.22, 0.30)},
    },
}

#: 对象（圆盘）允许的活动范围，单位 m（防止被推出"桌面"）。
_OBJECT_BOUNDS = {"x": (0.02, 0.90), "y_abs": (0.0, 0.48)}


def _sample_range(rng: np.random.Generator, rng_range: tuple[float, float]) -> float:
    """在闭区间内均匀采样一个标量（依据：附录 C 的均匀随机口径）。"""
    return float(rng.uniform(rng_range[0], rng_range[1]))


class BaseBimanualEnv:
    """双臂回合环境的公共骨架：关节伺服 + 噪声观测 + 回合计数。

    子类只需实现 :meth:`_reset_scene`（采样场景）与 :meth:`_task_dynamics`
    （对象演化）与 :meth:`_success_dist`（成功判据的距离量）。

    Attributes:
        params: :class:`dr.WorldParams` 本回合的世界参数（真值）。
        max_steps: 回合最大步数。
        joint_speed: 关节伺服限速，rad/步。
        _arm: :class:`arm.BimanualArm2D` 世界真值运动学。
        _rng: 回合随机数发生器（reset(seed) 重建 → 全回合确定）。
        _t: 已执行步数。
        _success: 是否已成功（成功后 done 保持 True）。
        _prev_dist: 上一步的成功判据距离（算进度奖励用），单位 m。
    """

    task_name: str = "base"

    def __init__(
        self,
        params: WorldParams,
        max_steps: int = MAX_STEPS,
        joint_speed: float = JOINT_SPEED,
    ) -> None:
        """保存世界参数与伺服配置（回合在 reset 时才真正开始）。

        Args:
            params: 世界参数（DR 采样结果，含真值连杆长/半径/摩擦等）。
            max_steps: 回合最大步数。
            joint_speed: 关节伺服限速，单位 rad/步。
        """
        self.params = params
        self.max_steps = int(max_steps)
        self.joint_speed = float(joint_speed)
        self._arm = BimanualArm2D(
            left_link=params.left_link, right_link=params.right_link, q0=HOME_Q
        )
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._success = False
        self._prev_dist = float("inf")
        # 场景状态（子类在 _reset_scene 里填充）。
        self.side = 1  # 对象/目标所在侧：+1 = 左半平面，-1 = 右半平面
        self.goal: np.ndarray = np.zeros((2, 2))  # reach: (2,2)；push/pick: (2,)
        self.object_pos: np.ndarray | None = None  # reach 任务为 None

    # ---------------------------------------------------------- 回合接口 --
    def reset(self, seed: int, split: str = "seen") -> dict:
        """重置回合：按 seed 采样场景，返回初始观测。

        Args:
            seed: 回合种子（int，非负）；决定场景采样与全部噪声序列。
            split: ``"seen"`` 或 ``"unseen"``——目标位姿采样范围的分层
                （评测协议的泛化轴，教程 §5.6 / §6.5）。

        Returns:
            obs：模块 docstring 表格约定的观测 dict。
        """
        if split not in ("seen", "unseen"):
            raise ValueError(f"split 须为 'seen'/'unseen'，当前 {split!r}")
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._success = False
        self._arm.set_state(HOME_Q)
        self.side = 1 if self._rng.uniform() < 0.5 else -1  # 左/右侧各半
        self._reset_scene(split)
        self._prev_dist = self._success_dist()
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """推进一步：伺服关节 → 演化对象 → 判成功 → 生成观测。

        Args:
            action: (5,) ``[qL1, qL2, qR1, qR2, grip]``；关节目标 rad、
                夹爪 <0.5 开 / ≥0.5 闭（reach/push 忽略夹爪位）。

        Returns:
            ``(obs, reward, done, info)``：进度奖励（距离减少量，成功步
            额外 +1）、是否结束（成功或超时）、info（``success / dist /
            t / ee_true`` 及任务特定字段，见子类）。
        """
        action = np.asarray(action, dtype=float).reshape(5)
        grip = float(action[4])
        # ① 低层关节伺服：动作缩放（控制接口 DR 维）乘在"目标−当前"增量
        # 上，再限幅（依据：位置伺服 + 接口增益失配的最小模型）。
        q = self._arm.q
        # 分关节限幅：q1 / q3 用 Q1_LIMITS，q2 / q4 用 Q2_LIMITS（两关节
        # 限位不同，不能统一裁到 Q1 区间）。
        q_target = np.asarray(action[:4], dtype=float).copy()
        q_target[0] = np.clip(q_target[0], *Q1_LIMITS)
        q_target[2] = np.clip(q_target[2], *Q1_LIMITS)
        q_target[1] = np.clip(q_target[1], *Q2_LIMITS)
        q_target[3] = np.clip(q_target[3], *Q2_LIMITS)
        q_eff = q + self.params.action_scale * (q_target - q)
        q_new = q + np.clip(q_eff - q, -self.joint_speed, self.joint_speed)
        ee_prev = self._arm.end_effectors()
        self._arm.set_state(q_new)
        ee_new = self._arm.end_effectors()
        ee_disp = ee_new - ee_prev  # (2,2) 每臂末端位移，m

        # ② 任务物理：对象演化（子类实现）。
        self._task_dynamics(grip, ee_prev, ee_new, ee_disp)

        # ③ 判据与奖励：进度 = 距离减少量；成功步再 +1（依据：稠密奖励的
        # 教学惯例，评分只看 info["success"]——协议口径只有成功率）。
        self._t += 1
        dist = self._success_dist()
        success = dist < self._tolerance() and self._success_gate()
        reward = (self._prev_dist - dist) + (1.0 if success else 0.0)
        self._prev_dist = dist
        self._success = self._success or success
        done = self._success or self._t >= self.max_steps
        info = self._info(success, dist)
        return self._observe(), float(reward), bool(done), info

    def _success_gate(self) -> bool:
        """成功的附加门条件（默认恒真；pick_place 用它实现"松开后才算
        放置"——拎着到目标不计成功）。"""
        return True

    # ---------------------------------------------------- 子类实现点 --
    def _reset_scene(self, split: str) -> None:
        """采样本回合场景（目标/对象初始位姿）。子类必须实现。"""
        raise NotImplementedError

    def _task_dynamics(
        self, grip: float, ee_prev: np.ndarray, ee_new: np.ndarray, ee_disp: np.ndarray
    ) -> None:
        """对象演化（无对象任务为 no-op）。子类按需实现。"""

    def _success_dist(self) -> float:
        """成功判据的距离量（单位 m）：reach 为双臂最大偏差，操作任务为
        对象到目标的距离。子类必须实现。"""
        raise NotImplementedError

    def _tolerance(self) -> float:
        """成功容差（单位 m）。子类必须实现。"""
        raise NotImplementedError

    def _info(self, success: bool, dist: float) -> dict:
        """公共 info 字段 + 任务特定字段（子类可覆盖扩展）。"""
        return {
            "success": bool(success),
            "dist": float(dist),
            "t": int(self._t),
            "ee_true": self._arm.end_effectors(),
        }

    # ---------------------------------------------------------- 观测 --
    def _observe(self) -> dict:
        """按模块 docstring 的表格构造带噪观测（噪声只加在感知字段）。

        噪声为 i.i.d. 高斯、标准差 ``params.obs_noise_std``（依据：DR 的
        "传感器"维）。目标与关节角不加噪声（目标 = 任务指令，关节角 =
        本体感知）。
        """
        sigma = self.params.obs_noise_std
        ee_true = self._arm.end_effectors()
        ee_obs = ee_true + self._rng.normal(0.0, sigma, size=(2, 2))
        radius = self.params.object_radius + float(self._rng.normal(0.0, sigma))
        if self.object_pos is None:
            object_obs = np.zeros(2)
        else:
            object_obs = self.object_pos + self._rng.normal(0.0, sigma, size=2)
        return {
            "q": self._arm.get_state(),
            "ee": ee_obs,
            "object": object_obs,
            "has_object": self.object_pos is not None,
            "radius": np.float64(radius),
            "goal": self.goal.copy(),
            "holding": np.bool_(getattr(self, "_held", False)),
        }


class ReachTask(BaseBimanualEnv):
    """reach：双臂末端各自到达目标点（教程 §2.2 双臂任务的"到达"基元）。

    成功 = ``max_s ‖ee_s − goal_s‖ < REACH_TOL``；无对象。
    info 额外字段：无（公共字段即可）。
    """

    task_name = "reach"

    def _reset_scene(self, split: str) -> None:
        """采样双臂目标点：左臂固定 y>0 一侧、右臂固定 y<0 一侧（各臂不
        交叉够远处目标——交叉位形在关节限位内不可行），范围见 SCENE_RANGES。
        （``side`` 只影响操作任务的"对象归哪条臂"，reach 不用它。）"""
        ranges = SCENE_RANGES["reach"][split]
        self.object_pos = None
        self.goal = np.zeros((2, 2))
        self.goal[LEFT] = [
            _sample_range(self._rng, ranges["x"]),
            _sample_range(self._rng, ranges["y_abs"]),
        ]
        self.goal[RIGHT] = [
            _sample_range(self._rng, ranges["x"]),
            -_sample_range(self._rng, ranges["y_abs"]),
        ]

    def _success_dist(self) -> float:
        # 双臂各自到目标的偏差取 max（木桶口径：有一臂没到即不算到）。
        return float(np.max(np.linalg.norm(self._arm.end_effectors() - self.goal, axis=1)))

    def _tolerance(self) -> float:
        return REACH_TOL


class PushTask(BaseBimanualEnv):
    """push：末端把圆盘推到目标点（接触-滑动摩擦模型）。

    接触判定：``‖ee_s − 圆心‖ ≤ R + PUSH_PAD``（任一臂）；接触时圆盘沿该
    臂末端位移方向平移 ``friction × |ee 位移|``（依据：准静态滑动的最简
    模型——位移传递系数即"摩擦"DR 维）。
    info 额外字段：``contact``（本步是否有臂接触圆盘）。
    """

    task_name = "push"

    def _reset_scene(self, split: str) -> None:
        """采样圆盘初始位姿（start 区）与目标点（seen/unseen 区）。"""
        start = SCENE_RANGES["push"]["start"]
        goal = SCENE_RANGES["push"][split]
        self.object_pos = np.array(
            [
                _sample_range(self._rng, start["x"]),
                self.side * _sample_range(self._rng, start["y_abs"]),
            ]
        )
        self.goal = np.array(
            [
                _sample_range(self._rng, goal["x"]),
                self.side * _sample_range(self._rng, goal["y_abs"]),
            ]
        )

    def _task_dynamics(self, grip, ee_prev, ee_new, ee_disp) -> None:
        """接触 → 摩擦滑动（沿接触法向 = "推方向"）。

        模型（依据：准静态推物的最简法向模型）：对每条臂，
        1. 接触判定 ``‖ee − 圆心‖ ≤ R + PUSH_PAD``；
        2. 推方向 = 接触法向 ``n = (圆心 − ee)/‖·‖``（从末端指向圆心）；
        3. 圆盘位移 = ``friction × max(⟨ee 位移, n⟩, 0) × n``——只有末端
           **迎着**圆盘压进去的位移分量才推动它（法向单侧接触：推得动、
           拉不动、切向滑过不拖拽——这就是"对象沿推方向移动"）。
        """
        assert self.object_pos is not None
        radius = self.params.object_radius
        contact = False
        for s in (LEFT, RIGHT):
            offset = self.object_pos - ee_new[s]
            gap = float(np.linalg.norm(offset))
            if gap <= radius + PUSH_PAD:
                contact = True
                normal = offset / max(gap, 1e-9)  # 推方向（接触法向）
                # 末端位移在推方向上的分量（只保留压入的那一侧）。
                push_amount = max(float(np.dot(ee_disp[s], normal)), 0.0)
                # 依据：摩擦滑动模型——位移 = friction × 法向推入量。
                self.object_pos = self.object_pos + self.params.friction * push_amount * normal
        # 圆盘不出"桌面"（夹取世界边界，防滑出后判据失真）。
        self.object_pos[0] = np.clip(self.object_pos[0], *_OBJECT_BOUNDS["x"])
        self.object_pos[1] = np.clip(
            self.object_pos[1], -_OBJECT_BOUNDS["y_abs"][1], _OBJECT_BOUNDS["y_abs"][1]
        )
        self._contact = contact

    def _success_dist(self) -> float:
        assert self.object_pos is not None
        return float(np.linalg.norm(self.object_pos - self.goal))

    def _tolerance(self) -> float:
        return PUSH_TOL

    def _info(self, success: bool, dist: float) -> dict:
        info = super()._info(success, dist)
        info["contact"] = bool(getattr(self, "_contact", False))
        return info


class PickPlaceTask(BaseBimanualEnv):
    """pick_place：对准 → 闭合吸附 → 搬运 → 松开放置（夹爪状态机）。

    状态机（``_held`` / ``_hold_offset`` / ``_hold_side``）：
    1. **open**：开爪接触不推动对象（"从上方接近"的 2D 化简）；
    2. **attached**：闭合爪（grip ≥ 0.5）且某臂 ``‖ee − 圆心‖ ≤ R +
       GRASP_PAD`` 时吸附——记录偏移 ``offset = object − ee``，此后对象
       随该臂刚体平移（"抓取后随动"）；
    3. **released**：grip < 0.5 时松开，对象停留原位。
    成功 = **松开后** ``‖object − goal‖ < PLACE_TOL``（拎着到目标不算，
    必须完成"放置"）。
    info 额外字段：``holding``、``holding_side``。
    """

    task_name = "pick_place"

    def _reset_scene(self, split: str) -> None:
        """采样对象初始位姿（start 区）与放置目标（seen/unseen 区）。"""
        start = SCENE_RANGES["pick_place"]["start"]
        goal = SCENE_RANGES["pick_place"][split]
        self.object_pos = np.array(
            [
                _sample_range(self._rng, start["x"]),
                self.side * _sample_range(self._rng, start["y_abs"]),
            ]
        )
        self.goal = np.array(
            [
                _sample_range(self._rng, goal["x"]),
                self.side * _sample_range(self._rng, goal["y_abs"]),
            ]
        )
        self._held = False
        self._hold_offset = np.zeros(2)
        self._hold_side = -1

    def _task_dynamics(self, grip, ee_prev, ee_new, ee_disp) -> None:
        """夹爪状态机推进（见类 docstring 的三状态）。"""
        assert self.object_pos is not None
        radius = self.params.object_radius
        if self._held:
            # attached：对象随持握臂刚体平移（偏移保持不变）。
            self.object_pos = ee_new[self._hold_side] + self._hold_offset
            if grip < 0.5:  # 松开：对象停留在当前位置
                self._held = False
                self._hold_side = -1
        else:
            if grip >= 0.5:
                # 吸附判定：闭合 + 接触（左臂先查，保证确定性）。
                for s in (LEFT, RIGHT):
                    if np.linalg.norm(ee_new[s] - self.object_pos) <= radius + GRASP_PAD:
                        self._held = True
                        self._hold_side = s
                        self._hold_offset = self.object_pos - ee_new[s]
                        break
        # 对象不出"桌面"。
        self.object_pos[0] = np.clip(self.object_pos[0], *_OBJECT_BOUNDS["x"])
        self.object_pos[1] = np.clip(
            self.object_pos[1], -_OBJECT_BOUNDS["y_abs"][1], _OBJECT_BOUNDS["y_abs"][1]
        )

    def _success_dist(self) -> float:
        assert self.object_pos is not None
        return float(np.linalg.norm(self.object_pos - self.goal))

    def _success_gate(self) -> bool:
        # 必须松开（released 状态）才允许判成功：拎着到目标不算放置。
        return not self._held

    def _tolerance(self) -> float:
        return PLACE_TOL

    def _info(self, success: bool, dist: float) -> dict:
        info = super()._info(success, dist)
        info["holding"] = bool(self._held)
        info["holding_side"] = int(self._hold_side)
        return info


def make_task(name: str, params: WorldParams, **kwargs) -> BaseBimanualEnv:
    """按任务名构造回合环境（benchmark 的唯一入口）。

    Args:
        name: 任务名，``"reach" / "push" / "pick_place"``。
        params: 世界参数（DR 采样结果）。
        **kwargs: 透传给环境构造器（``max_steps`` / ``joint_speed``）。

    Returns:
        :class:`BaseBimanualEnv` 对应子类实例。

    Raises:
        ValueError: 任务名不认识时抛出（列出可用任务）。
    """
    classes = {"reach": ReachTask, "push": PushTask, "pick_place": PickPlaceTask}
    if name not in classes:
        raise ValueError(f"未知任务 {name!r}；可用：{sorted(classes)}")
    return classes[name](params, **kwargs)
