"""两种脚本化对照策略（calibrated / robust）——剂量对照实验的"被评测者"。

函数流水线
----------
本模块是被 ``benchmark.run`` 逐回合驱动的策略层：``reset(task)`` 标记回合
开始，``act(obs, step) -> action (5,)`` 每步产出关节目标 + 夹爪位。
策略内部只依赖 :mod:`arm` 的解析 IK 与自己的"模型世界"（连杆长/对象
半径），不读 :class:`dr.WorldParams` 真值——**模型与世界的差就是被研究的
失配**。依赖方向：``policies → arm``（FK/IK）、``policies → dr``（仅取
名义常数做"CAD 模型"）。

两种策略的设计（对照实验的自变量只有一个）
------------------------------------------
- :class:`CalibratedPolicy`：**用名义模型**（CAD 连杆长 + CAD 对象半径）
  规划全部技能。标定世界（``none`` 档）里它即完美；DR 一开，世界连杆长
  偏离 CAD，它的"IK → 实际末端"映射出现系统性偏移（≈ (s−1)×工作半径，
  s 为臂长缩放），剂量越大偏移越大——成功率随 DR 剂量单调下降。这复现
  教程 §6.4 的核心论断：只有"干净/标定分布"经验的策略，在随机化条件下
  脆弱。
- :class:`RobustPolicy`：**每回合先测量再行动**（自标定）：前 4 步向
  探测位形机动，用带噪观测 (q, ee) 对做最小二乘估计两节连杆长，并把
  ``radius`` 读数平均成对象半径估计，之后所有 IK 用测量值。对臂长/半径
  DR 稳健——对应教程 §4.2"让真实条件落进策略'见过'的分布"的 2D 版：
  与其记住一个模型，不如把模型当作待估参数。

两种策略共享同一套技能骨架（reach 一回合一次规划；push 逐步闭环；
pick_place 四阶段状态机），因此成功率差异**只能**来自模型失配 vs 自标定
——这正是对照实验要的单变量性。真策略（VLA 训练）不在本项目范围，
见教程第 08 章。

动作格式（与 ``tasks.py`` 的回合接口一致）
------------------------------------------
``action = (5,) = [qL1, qL2, qR1, qR2, grip]``：前 4 项关节目标（rad，
低层限速伺服会跟踪），第 5 项夹爪（< 0.5 开 / ≥ 0.5 闭）。
"""
from __future__ import annotations

import numpy as np

from arm import BASES, LEFT, RIGHT, link_ik
from dr import (
    NOMINAL_LINK_LEFT,
    NOMINAL_LINK_RIGHT,
    NOMINAL_RADIUS,
    PHYS_BOUNDS,
)
from tasks import PLACE_TOL, PUSH_TOL

__all__ = ["CalibratedPolicy", "RobustPolicy", "PROBE_Q"]

#: robust 策略的探测位形（与 home (0.9, -1.8) 拉开距离，保证两次读数的
#: 几何方向差异足够大 → 最小二乘良态）。
PROBE_Q: np.ndarray = np.array([0.45, -0.9, 0.45, -0.9])
#: robust 策略的测量步数（home + 3 个中途读数 = 4 组 (q, ee) 观测）。
_MEASURE_STEPS: int = 4


def _hold_action(obs: dict, grip: float = 0.0) -> np.ndarray:
    """"原地保持"动作：以当前关节角为目标（伺服不再移动），爪位 grip。"""
    return np.concatenate([np.asarray(obs["q"], dtype=float).reshape(4), [grip]])


class ScriptedPolicy:
    """脚本策略骨架：技能状态机 + "模型世界"钩子（子类只换模型来源）。

    Attributes:
        _task: 当前任务名（``reset`` 时写入）。
        _model_links: ((L1,L2), (L1,L2)) 策略自认为的双臂连杆长，单位 m
            （calibrated = 名义值；robust = 回合内测量值）。
        _model_radius: 策略自认为的对象半径，单位 m。
        _plan: reach 任务缓存的关节目标 (4,)（一回合一次规划）。
        _phase: pick_place 状态机当前阶段。
    """

    def __init__(self) -> None:
        self._task: str = ""
        self._model_links: tuple[tuple[float, float], tuple[float, float]] = (
            tuple(NOMINAL_LINK_LEFT),
            tuple(NOMINAL_LINK_RIGHT),
        )
        self._model_radius: float = NOMINAL_RADIUS
        self._plan: np.ndarray | None = None
        self._phase: str = ""
        # push 卡滞检测状态（历史最优距离 + 距上次改善的步数）。
        self._best_dist: float = float("inf")
        self._since_improve: int = 0

    # ----------------------------------------------------- 生命周期 --
    def reset(self, task: str) -> None:
        """开始新回合：清空缓存的规划与状态机阶段。

        Args:
            task: 任务名（``"reach" / "push" / "pick_place"``）。
        """
        self._task = task
        self._plan = None
        self._phase = ""
        self._best_dist = float("inf")
        self._since_improve = 0

    def act(self, obs: dict, step: int) -> np.ndarray:
        """每步动作入口：按任务分发到对应技能。

        Args:
            obs: 环境观测（见 ``tasks`` 模块的观测表）。
            step: 回合内步数计数（从 0 开始）。

        Returns:
            (5,) 动作 ``[qL1, qL2, qR1, qR2, grip]``。
        """
        if self._task == "reach":
            return self._act_reach(obs)
        if self._task == "push":
            return self._act_push(obs)
        if self._task == "pick_place":
            return self._act_pick(obs)
        raise ValueError(f"未知任务 {self._task!r}")

    # --------------------------------------------------- 技能：reach --
    def _act_reach(self, obs: dict) -> np.ndarray:
        """reach 技能：一回合规划一次——双臂目标各解一次 IK 后反复下发
        （低层限速伺服负责收敛；规划本身开环，模型失准直接表现为末端
        停偏——这正是 calibrated 在 DR 下失败的位置）。"""
        if self._plan is None:
            goals = np.asarray(obs["goal"], dtype=float).reshape(2, 2)
            q_l = self._ik(LEFT, goals[LEFT])
            q_r = self._ik(RIGHT, goals[RIGHT])
            # IK 无解（目标超模型工作空间）→ 原地保持（协议记失败）。
            q_l = self._arm_q(obs, LEFT) if q_l is None else q_l
            q_r = self._arm_q(obs, RIGHT) if q_r is None else q_r
            self._plan = np.concatenate([q_l, q_r])
        return np.concatenate([self._plan, [0.0]])

    # ---------------------------------------------------- 技能：push --
    def _act_push(self, obs: dict) -> np.ndarray:
        """push 技能（三段进近 + 带前馈的推土机跟随，逐步闭环）：

        1. ``side``：先到**侧向安全点** ``对象 + 垂直推方向×(R̂+0.10)``
           （绕开圆盘，防进近途中把它撞歪；偏移用**模型**半径）；
        2. ``back``：退到圆盘正后方 ``对象 − 推方向×2R̂``；
        3. ``push``：**速度级**推土机跟随——命令点 = 当前末端 + march·推
           方向 + 0.3·横向回中（始终压住圆盘背面、把圆盘往目标线上引），
           且沿推方向不得超过"圆心后 0.2R"；距目标 < 0.5·PUSH_TOL 后原地
           保持（防抖动推出容差）。圆盘每步位移 ≤ march ≪ 容差 → 不会
           一步跨过成功窗口。

        闭环只反馈"对象位置"（每步重读 obs），关节指令仍经模型 IK 生成
        ——闭环修不掉的模型失配（工作半径整体缩放）依然伤害本技能，
        只是伤害小于开环的 reach。
        """
        obj = np.asarray(obs["object"], dtype=float).reshape(2)
        goal = np.asarray(obs["goal"], dtype=float).reshape(2)
        side = LEFT if obj[1] >= 0 else RIGHT  # 对象在哪半平面就归哪条臂
        idle = _hold_action(obs)
        delta = goal - obj
        dist = float(np.linalg.norm(delta))
        if dist < 0.5 * PUSH_TOL:  # 已到位：原地保持
            return idle
        direction = delta / max(dist, 1e-9)
        ee = np.asarray(obs["ee"], dtype=float).reshape(2, 2)[side]
        if self._phase == "":
            self._phase = "side"
        if self._phase in ("side", "back"):
            sign = 1.0 if side == LEFT else -1.0
            if self._phase == "side":
                # 垂直推方向、指向对象外侧（远离桌面中线）的绕行点。
                perp = sign * np.array([-direction[1], direction[0]])
                waypoint = obj + perp * (self._model_radius + 0.10)
                next_phase = "back"
            else:
                waypoint = obj - direction * 2.0 * self._model_radius
                next_phase = "push"
            if float(np.linalg.norm(ee - waypoint)) < 0.02:
                self._phase = next_phase
            else:
                q = self._ik(side, waypoint)
                if q is None:
                    return idle
                action = idle.copy()
                action[2 * side: 2 * side + 2] = q
                return action
        # push 阶段：**速度级**指令——命令点 = 当前末端 + march·推方向 +
        # 0.3·横向回中项，且沿推方向不得超过"圆心后 0.2R"（防越过中心
        # 反向推）。速度级的好处：圆盘每步位移 ≤ f·march ≪ 成功容差，
        # 不会一步跨过目标窗口；f=1 时命令点恒在末端前方 march 处，
        # 不会静止按压（位移传递物理要求"推者不停"）。
        # 卡滞检测（窗口式，抗观测噪声）：若"到目标的距离"连续 10 步没有
        # 至少 0.005 的改善，判定推送预算耗尽（低摩擦下末端被 0.2R 钳位
        # 拖住）→ 退回 back 相位重新建立推送间隙（推土机的"倒一步再推"；
        # 重进近跳过 side 绕行——此时末端本就在圆盘后方，沿直线退即安全）。
        if dist < self._best_dist - 0.005:
            self._best_dist = dist
            self._since_improve = 0
        else:
            self._since_improve += 1
        if self._since_improve > 10:
            self._since_improve = 0
            self._best_dist = dist
            self._phase = "back"
            return self._act_push(obs)
        radius = max(self._model_radius, 1e-3)
        march = min(0.04, 0.5 * radius)
        gap_vec = obj - ee  # 末端 → 圆心
        along = gap_vec - float(np.dot(gap_vec, direction)) * direction
        cmd = ee + march * direction + 0.3 * along
        excess = float(np.dot(cmd - obj, direction)) + 0.2 * radius
        if excess > 0:  # 命令点太靠前 → 拉回圆心后方 0.2R
            cmd = cmd - excess * direction
        q = self._ik(side, cmd)
        if q is None:
            return idle
        action = idle.copy()
        action[2 * side: 2 * side + 2] = q
        return action

    # ----------------------------------------------- 技能：pick_place --
    def _act_pick(self, obs: dict) -> np.ndarray:
        """pick_place 技能：四阶段状态机（对应 3D 的"经上方→闭合→搬运→
        松开"的 2D 化简）：

        1. ``approach``：开爪进近到圆心近侧 40% 半径处（防撞歪对象）；
        2. ``grasp``：进到圆心并闭合（吸附由环境接触判定决定）；
        3. ``carry``：闭合搬运到目标点上方（IK 到目标）；
        4. ``release``：松开（环境判定放置成功）。
        """
        obj = np.asarray(obs["object"], dtype=float).reshape(2)
        goal = np.asarray(obs["goal"], dtype=float).reshape(2)
        side = LEFT if obj[1] >= 0 else RIGHT
        idle = _hold_action(obs)
        radius = max(self._model_radius, 1e-3)

        if self._phase == "":
            self._phase = "approach"
        if self._phase == "approach":
            # 近侧停驻点：圆心沿"肩→对象"方向回退 40% 半径。
            to_obj = obj - np.asarray(BASES[side], dtype=float)
            u = to_obj / max(float(np.linalg.norm(to_obj)), 1e-9)
            target_pt = obj - 0.4 * radius * u
            ee = np.asarray(obs["ee"], dtype=float).reshape(2, 2)[side]
            if float(np.linalg.norm(ee - obj)) < 1.5 * radius:
                self._phase = "grasp"  # 已在抓取窗边缘 → 进入闭合阶段
            q = self._ik(side, target_pt)
            if q is None:
                return idle
            action = idle.copy()
            action[2 * side: 2 * side + 2] = q
            return action
        if self._phase == "grasp":
            if bool(np.asarray(obs["holding"])):
                self._phase = "carry"
                return self._act_pick(obs)  # 立即转入搬运（不浪费步数）
            q = self._ik(side, obj)  # 进到圆心，爪闭合
            if q is None:
                return idle
            action = idle.copy()
            action[2 * side: 2 * side + 2] = q
            action[4] = 1.0
            return action
        if self._phase == "carry":
            # 搬运：吸附偏移 offset = 对象 − 末端 在持握中保持不变，因此
            # 要把**对象**送到目标，末端应指向 goal − offset（用带噪读数
            # 逐步估计）。松开条件直接看对象读数与目标的距离。
            ee = np.asarray(obs["ee"], dtype=float).reshape(2, 2)[side]
            offset = obj - ee
            if float(np.linalg.norm(obj - goal)) < 0.6 * PLACE_TOL:
                self._phase = "release"
            else:
                q = self._ik(side, goal - offset)
                if q is not None:
                    action = idle.copy()
                    action[2 * side: 2 * side + 2] = q
                    action[4] = 1.0
                    return action
            # IK 无解 → 落入松开（放在当前位置，协议判失败）
        # release：松开一爪，之后原地保持。
        action = idle.copy()
        action[4] = 0.0
        return action

    # -------------------------------------------------------- 工具 --
    def _ik(self, side: int, target: np.ndarray) -> np.ndarray | None:
        """用**模型**连杆长解 IK（失配的唯一入口：模型错 → 关节目标错）。

        Args:
            side: 臂侧索引（:data:`arm.LEFT` / :data:`arm.RIGHT`）。
            target: (2,) 期望末端位置，单位 m（桌面系）。

        Returns:
            (2,) 关节目标；模型工作空间外为 ``None``。
        """
        return link_ik(self._model_links[side], target, BASES[side], elbow=-1.0)

    @staticmethod
    def _arm_q(obs: dict, side: int) -> np.ndarray:
        """从观测取某一臂当前关节角 (2,)（IK 失败时的"保持"目标）。"""
        return np.asarray(obs["q"], dtype=float).reshape(4)[2 * side: 2 * side + 2]


class CalibratedPolicy(ScriptedPolicy):
    """calibrated：信任名义 CAD 模型的脚本策略（对照组）。

    模型 = 名义连杆长 / 名义对象半径（:mod:`dr` 的 ``NOMINAL_*`` 常数），
    回合内从不更新——标定世界（none 档）里即完美，DR 档位越高失准越重。
    教学对应：教程 §4.2.1 ① "用干净/标定数据训练的策略遇到没见过的条件
    就失效"的脚本策略版。
    """


class RobustPolicy(ScriptedPolicy):
    """robust：每回合先用带噪观测自标定，再执行同一套技能（实验组）。

    测量内容（每回合一次，占前 :data:`_MEASURE_STEPS` 步）：
    - **连杆长**：向 :data:`PROBE_Q` 机动，收集 home + 中途共 4 组
      ``(q, ee)`` 读数，对每组按 FK 线性性列方程
      ``[c1, c12; s1, s12]·[L1, L2] = ee − base``，4 组堆成 (8, 2) 最小
      二乘（依据：FK 对 (L1, L2) 严格线性，故测量是线性回归而非非线性
      拟合——这是选 2 连杆做教学本体的原因之一）；
    - **对象半径**：``radius`` 读数取均值（读数本身已含观测噪声）。
    之后所有 IK 用测量值 → 对臂长/半径 DR 稳健。测量精度受观测噪声
    限制（strong 档 ~0.01 m 级），但远小于 calibrated 的模型失配
    （strong 档 ~0.05-0.1 m 级）——这正是两曲线分叉的量化来源。

    Attributes:
        measured_links: ((L1,L2),(L1,L2)) 最近一次回合的测量连杆长
            （测试与 DEBUG 用；未测量时为 None）。
        measured_radius: 最近一次回合的测量对象半径（同上）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.measured_links: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.measured_radius: float | None = None
        self._probe_obs: list[dict] = []

    def reset(self, task: str) -> None:
        """新回合：清空测量缓存（上一回合的测量作废——世界可能重采）。"""
        super().reset(task)
        self._probe_obs = []

    def act(self, obs: dict, step: int) -> np.ndarray:
        """前 ``_MEASURE_STEPS`` 步做探测测量（step 0..3），之后走技能。

        Args:
            obs: 环境观测。
            step: 回合内步数计数。

        Returns:
            (5,) 动作；测量期为探测机动指令。
        """
        if step < _MEASURE_STEPS:
            # 缓存带噪读数（浅拷贝数组字段，防环境就地修改——实际环境
            # 每步新建 obs，这里按防御式约定处理）。
            self._probe_obs.append(
                {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
            )
            if step == _MEASURE_STEPS - 1:
                self._measure()
            # 探测机动：向 PROBE_Q 限速移动（与伺服一致的逻辑目标）。
            return np.concatenate([PROBE_Q, [0.0]])
        return super().act(obs, step)

    def _measure(self) -> None:
        """由缓存的 (q, ee) 读数做线性最小二乘：估计双臂连杆长 + 对象半径。

        每条臂的 FK 可写成 ``ee − base = M(q)·[L1, L2]``，其中
        ``M(q) = [[c1, c12], [s1, s12]]``——对连杆长**线性**。4 组读数堆叠
        为 (8, 2) 超定方程，``np.linalg.lstsq`` 最小二乘求解；半径取读数
        均值。估计值截断到物理界（防偶发坏读数产生非法模型）。
        """
        assert len(self._probe_obs) == _MEASURE_STEPS
        # 一次解双臂：把 (L1,L2)[左] 与 (L1,L2)[右] 排成 4 参数向量。
        # 每条 (读数, 臂) 贡献两行方程（x 行用 [c1, c12]，y 行用 [s1, s12]，
        # 只写入该臂对应的参数块）——FK 对连杆长线性，故这是普通线性 LS。
        n_rows = 2 * 2 * len(self._probe_obs)
        a_mat = np.zeros((n_rows, 4))
        b_vec = np.zeros(n_rows)
        r_idx = 0
        for reading in self._probe_obs:
            q = np.asarray(reading["q"], dtype=float).reshape(4)
            ee = np.asarray(reading["ee"], dtype=float).reshape(2, 2)
            for side in (LEFT, RIGHT):
                q_s = q[2 * side: 2 * side + 2]
                c1, s1 = np.cos(q_s[0]), np.sin(q_s[0])
                c12, s12 = np.cos(q_s[0] + q_s[1]), np.sin(q_s[0] + q_s[1])
                cols = slice(2 * side, 2 * side + 2)
                base = np.asarray(BASES[side], dtype=float)
                a_mat[r_idx, cols] = [c1, c12]
                b_vec[r_idx] = ee[side][0] - base[0]
                r_idx += 1
                a_mat[r_idx, cols] = [s1, s12]
                b_vec[r_idx] = ee[side][1] - base[1]
                r_idx += 1
        sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        lo, hi = PHYS_BOUNDS["link"]
        clamp = lambda v: float(min(max(v, lo), hi))  # noqa: E731
        self._model_links = (
            (clamp(sol[0]), clamp(sol[1])),
            (clamp(sol[2]), clamp(sol[3])),
        )
        self.measured_links = self._model_links
        radius_readings = [
            float(np.asarray(r["radius"])) for r in self._probe_obs
        ]
        r_lo, r_hi = PHYS_BOUNDS["object_radius"]
        self._model_radius = float(
            min(max(float(np.mean(radius_readings)), r_lo), r_hi)
        )
        self.measured_radius = self._model_radius
