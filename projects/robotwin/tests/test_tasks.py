"""projects/robotwin/tasks 的测试 —— 直接运行：``python tests/test_tasks.py``。

验证三种双臂任务回合环境（reach / push / pick_place）的回合接口与物理：
同 seed 逐位确定、解析 IK 策略在标定世界必成功、push 的接触判定/推方向/
摩擦缩放、pick_place 夹爪状态机（未抓取不动 / 抓取随动 / 松开才算放置）、
超时终止与 info 字段。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tasks as tk  # noqa: E402
from arm import LEFT, link_ik  # noqa: E402
from arm import BASES  # noqa: E402
from dr import WorldParams  # noqa: E402

NOMINAL = WorldParams.nominal()


def _ideal_reach_action(env, obs):
    """理想 reach 动作：用**环境真值**连杆长解 IK（测试专用——不是
    policies 里的策略，那里故意只用模型连杆长）。"""
    goal = np.asarray(obs["goal"]).reshape(2, 2)
    q_l = link_ik(env.params.left_link, goal[0], BASES[0])
    q_r = link_ik(env.params.right_link, goal[1], BASES[1])
    assert q_l is not None and q_r is not None  # 场景范围保证可达
    return np.concatenate([q_l, q_r, [0.0]])


def test_reset_same_seed_bitwise_identical():
    """同 seed 重置 → 观测逐位一致（含噪声字段），三个任务皆然。"""
    for name in tk.TASKS:
        env_a = tk.make_task(name, NOMINAL)
        env_b = tk.make_task(name, NOMINAL)
        obs_a = env_a.reset(seed=42)
        obs_b = env_b.reset(seed=42)
        for key in obs_a:
            va, vb = obs_a[key], obs_b[key]
            if isinstance(va, np.ndarray):
                assert np.array_equal(va, vb), (name, key)
            else:
                assert va == vb, (name, key)
    # 换 seed 必须不同（否则"确定性"只是"恒定"）。
    env_c = tk.make_task("push", NOMINAL)
    assert not np.array_equal(env_c.reset(seed=1)["object"], env_c.reset(seed=2)["object"])


def test_rollout_deterministic_same_seed():
    """同 seed + 同动作序列 → 整条 rollout 的 (obs, reward, done) 逐位一致。"""
    actions = [np.array([0.9, -1.2, 0.9, -1.2, 0.0]) + 0.01 * k for k in range(20)]
    runs = []
    for _ in range(2):
        env = tk.make_task("pick_place", NOMINAL)
        obs = env.reset(seed=7)
        trace = []
        for a in actions:
            obs, reward, done, info = env.step(a)
            trace.append((obs["q"].copy(), obs["ee"].copy(), obs["object"].copy(),
                          reward, done, info["dist"]))
        runs.append(trace)
    for step_a, step_b in zip(*runs):
        for va, vb in zip(step_a, step_b):
            if isinstance(va, np.ndarray):
                assert np.array_equal(va, vb)
            else:
                assert va == vb


def test_reach_solved_by_exact_ik_policy():
    """reach：从 home 出发、用环境真值连杆长的解析 IK 策略 → 必成功。

    12 个确定性种子全过（门限理由：标定世界 + 精确模型 = 零模型误差，
    唯一残余是伺服收敛性——限速 0.15 rad/步、horizon 64 步绰绰有余）。"""
    n_ok = 0
    n_total = 12
    for seed in range(n_total):
        env = tk.make_task("reach", NOMINAL)
        obs = env.reset(seed=100 + seed)
        action = _ideal_reach_action(env, obs)
        done = False
        while not done:
            obs, _r, done, info = env.step(action)
        assert info["success"], (seed, info["dist"])
        n_ok += 1
    print(f"\n[reach-ideal-ik] {n_ok}/{n_total} seeds success")


def test_push_contact_requires_proximity():
    """push 接触判定：末端离圆盘远 → 不接触、对象不动；末端压到圆心 →
    接触为真、对象开始位移。"""
    env = tk.make_task("push", NOMINAL)
    obs = env.reset(seed=5)
    obj0 = env.object_pos.copy()
    side = LEFT if obj0[1] >= 0 else 1
    # 阶段 1：末端停在 home 附近（离圆盘远）——对象必须纹丝不动。
    idle = np.concatenate([obs["q"], [0.0]])
    for _ in range(10):
        obs, _r, done, info = env.step(idle)
        assert not info["contact"]
        assert np.array_equal(env.object_pos, obj0)  # 位移为零
    # 阶段 2：末端向圆盘中心推进 → 接触建立、对象被推动。
    q_push = link_ik(env.params.left_link if side == 0 else env.params.right_link,
                     obj0, BASES[side])
    for _ in range(30):
        action = np.concatenate([obs["q"][: 2 * side], q_push,
                                 obs["q"][2 * side + 2:], [0.0]])
        obs, _r, done, info = env.step(action)
        if info["contact"]:
            break
    assert info["contact"], "末端到达圆心仍无接触 → 接触判定坏"
    assert np.linalg.norm(env.object_pos - obj0) > 0.0


def test_push_displacement_direction_and_friction():
    """push 位移物理两条性质：
    1. **方向**：直线航点推的净位移与推方向（+x）几乎共线（夹角余弦
       > 0.99——门限理由：侧向绕行 + 贴线推进后，残余弯曲只来自 IK
       关节插值的二阶效应）；
    2. **摩擦缩放**：单步接触传递位移 = f × 末端法向推入量——取末端
       恰在接触缘外 0.02 m 处走一步，f=1 与 f=0.5 的对象位移之比 ≈ 0.5
       （容差 ±0.1；注意长距离推送受"间隙预算"上限约束，不与 f 成简单
       比例，故此性质用单步传递检验）。"""
    def make_env(friction):
        params = WorldParams(NOMINAL.left_link, NOMINAL.right_link,
                             NOMINAL.object_radius, friction,
                             NOMINAL.obs_noise_std, NOMINAL.action_scale)
        return tk.make_task("push", params)

    def step_to(env, obs, side, q_t, n_steps=1):
        for _ in range(n_steps):
            action = np.concatenate([obs["q"][: 2 * side], q_t,
                                     obs["q"][2 * side + 2:], [0.0]])
            obs, _r, done, _i = env.step(action)
        return obs

    # ---- 性质 1：直线航点推的方向 ----
    env = make_env(1.0)
    obs = env.reset(seed=9)
    obj0 = env.object_pos.copy()
    side = 0 if obj0[1] >= 0 else 1
    link = env.params.left_link if side == 0 else env.params.right_link
    radius = env.params.object_radius
    side_sign = 1.0 if side == 0 else -1.0
    q_side = link_ik(link, obj0 + np.array([0.0, side_sign * (radius + 0.10)]),
                     BASES[side])
    obs = step_to(env, obs, side, q_side, n_steps=12)
    for xt in np.linspace(obj0[0] - 0.12, obj0[0] + 0.30, 22):
        obs = step_to(env, obs, side,
                      link_ik(link, np.array([xt, obj0[1]]), BASES[side]), n_steps=2)
    disp = env.object_pos - obj0
    cos = disp[0] / np.linalg.norm(disp)
    assert cos > 0.99, (cos, disp)
    print(f"\n[push-physics] straight-line disp = {disp.round(4)} (cos={cos:.4f})")

    # ---- 性质 2：单步摩擦传递比例 ----
    def single_step_disp(friction):
        env_f = make_env(friction)
        obs_f = env_f.reset(seed=9)  # 同 seed → 同场景
        obj_f = env_f.object_pos.copy()
        side_f = 0 if obj_f[1] >= 0 else 1
        link_f = env_f.params.left_link if side_f == 0 else env_f.params.right_link
        sign_f = 1.0 if side_f == 0 else -1.0
        # 三段无接触进近（各段与圆盘的最小距离 ≥ 0.075 > R + PUSH_PAD）：
        # 侧点 → 横移到正后方外侧 → 下降到"接触缘外 0.075 m"处。
        def q_of(point):
            return link_ik(link_f, point, BASES[side_f])

        obs_f = step_to(env_f, obs_f, side_f,
                        q_of(obj_f + np.array([0.0, sign_f * 0.15])), n_steps=12)
        obs_f = step_to(env_f, obs_f, side_f,
                        q_of(obj_f + np.array([-0.075, sign_f * 0.15])), n_steps=8)
        obs_f = step_to(env_f, obs_f, side_f,
                        q_of(obj_f + np.array([-0.075, 0.0])), n_steps=10)
        assert np.array_equal(env_f.object_pos, obj_f)  # 全程未接触 → 不动
        # 向圆心方向走 0.02 m（跨过接触缘 0.062）→ 一步传递。
        obs_f = step_to(env_f, obs_f, side_f,
                        q_of(obj_f + np.array([-0.055, 0.0])), n_steps=1)
        assert env_f._contact  # 已接触
        return env_f.object_pos - obj_f

    d_fast = single_step_disp(1.0)
    d_slow = single_step_disp(0.5)
    assert abs(d_fast[0] - 0.02) < 0.005, d_fast   # f=1：1:1 传递
    assert abs(d_slow[0] / d_fast[0] - 0.5) < 0.1, (d_fast, d_slow)
    print(f"[push-physics] single-step disp f=1: {d_fast[0]:.4f} m, "
          f"f=0.5: {d_slow[0]:.4f} m (ratio {d_slow[0] / d_fast[0]:.3f})")


def test_pick_place_state_machine():
    """pick_place 状态机三定律：
    1. 开爪接触不推动对象（未抓取时对象不动）；
    2. 闭合 + 接触 → 吸附，对象随末端刚体平移（偏移不变）；
    3. 拎到目标不算成功，松开后对象停在原地才判成功。"""
    env = tk.make_task("pick_place", NOMINAL)
    obs = env.reset(seed=11)
    obj0 = env.object_pos.copy()
    side = 0 if obj0[1] >= 0 else 1
    link = env.params.left_link if side == 0 else env.params.right_link
    q_center = link_ik(link, obj0, BASES[side])

    def action_for(q, grip):
        act = obs["q"].copy()
        act[2 * side: 2 * side + 2] = q
        return np.concatenate([act, [grip]])

    # ① 开爪压到圆心：对象必须不动。
    for _ in range(30):
        obs, _r, done, info = env.step(action_for(q_center, 0.0))
    assert not info["holding"]
    assert np.array_equal(env.object_pos, obj0)
    # ② 闭合 → 吸附；此后对象随末端平移（偏移恒定）。
    for _ in range(6):
        obs, _r, done, info = env.step(action_for(q_center, 1.0))
        if info["holding"]:
            break
    assert info["holding"], "圆心处闭合仍未吸附 → 接触判定坏"
    offset = env.object_pos - info["ee_true"][side]
    q_away = link_ik(link, env.goal + np.array([0.1, 0.05]), BASES[side])
    for _ in range(20):
        obs, _r, done, info = env.step(action_for(q_away, 1.0))
        assert info["holding"]  # 搬运中保持吸附
        assert np.allclose(env.object_pos - info["ee_true"][side], offset, atol=1e-12)
        assert not info["success"]  # ③ 拎着不算成功（即使已在目标附近）
    # ③ 松开 → 对象停在原地；当前位置即判成功/失败。
    dist_at_release_expected = np.linalg.norm(env.object_pos - env.goal)
    obs, _r, done, info = env.step(np.concatenate([obs["q"], [0.0]]))
    assert not info["holding"]
    pos_after_release = env.object_pos.copy()
    for _ in range(5):
        obs, _r, done, info = env.step(np.concatenate([obs["q"], [0.0]]))
        assert np.array_equal(env.object_pos, pos_after_release)  # 松开后静止
    assert info["success"] == (dist_at_release_expected < tk.PLACE_TOL)
    print(f"\n[pick-state-machine] release dist = {dist_at_release_expected:.3f} m"
          f" → success={info['success']}")


def test_horizon_termination_and_info():
    """超时终止：全程保持 home → horizon 步后 done=True 且未成功；
    info 含协议约定字段；朝目标推进时进度奖励为正。"""
    env = tk.make_task("reach", NOMINAL)
    obs = env.reset(seed=3)
    idle = np.concatenate([obs["q"], [0.0]])
    steps = 0
    while True:
        obs, reward, done, info = env.step(idle)
        steps += 1
        if done:
            break
    assert steps == tk.MAX_STEPS  # 超时（而非成功）终止
    assert not info["success"]
    for key in ("success", "dist", "t", "ee_true"):
        assert key in info
    # 进度奖励：向目标推进的第一步 reward > 0（距离减少量）。
    env2 = tk.make_task("reach", NOMINAL)
    obs2 = env2.reset(seed=3)
    action = _ideal_reach_action(env2, obs2)
    _obs, reward, _done, _info = env2.step(action)
    assert reward > 0.0, reward


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
