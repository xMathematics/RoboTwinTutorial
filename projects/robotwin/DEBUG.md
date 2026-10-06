# RoboTwin 迷你基准调试与测试教程（DEBUG）

> 零基础基准撰写。姊妹文档：[TUTORIAL.md](TUTORIAL.md)（代码导读）｜
> [METRICS.md](METRICS.md)（指标健康值——本文变量表的健康值与它保持一致）｜
> [.vscode/SETUP.md](../../.vscode/SETUP.md)（VS Code 配置总教程）。
> 规范出处：[CONSTRAINTS.md §4.5](../../CONSTRAINTS.md)。

---

## 1. 环境与两种测试

**解释器**：本项目纯 numpy，两套解释器均验证通过（35/35）——

| 解释器 | numpy | 全套件耗时（实测） | demo.py 耗时 |
|--------|-------|--------------------|--------------|
| conda `llm_env`（`~/anaconda3/envs/llm_env/bin/python`） | 2.2.6 | pytest ~3.8 s / 直跑 ~5.4 s | ~2.8 s |
| 系统 `python3` | 1.26.4 | 直跑 ~4.4 s | ~2.5 s |

两套解释器的 `demo.py` 输出**逐位一致**（有 diff 验证）；唯一版本敏感点是
robust 自标定里的 `np.linalg.lstsq`（SVD 实现，numpy 1.26 与 2.x 有 ~1e-12
量级差异）——远离一切断言门限，可忽略。

### 1.1 单点测试（改了一个模块 → 用它）

**方式 A：直跑过滤**（每个测试文件的 `__main__` 块都内置，推荐）：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/robotwin     # 必须在本目录（原因见 §1.4）

python tests/test_policies.py dose       # 只跑名字含 "dose" 的测试
# 预期输出：
# [calibrated dose curve] none=1.00, mild=0.85, strong=0.34
# PASS test_calibrated_dose_monotone_decrease
# 1/1 tests passed.
python tests/test_policies.py strong     # 前缀更宽：strong 相关测试命中
python tests/test_arm.py zzz             # 子串无任何匹配：列出全部可用测试名再退出（exit 1）
python tests/test_tasks.py               # 不带参数 = 该文件全部测试
```

**方式 B：pytest 过滤**：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/robotwin
pytest tests/test_tasks.py::test_pick_place_state_machine -v      # 精确到函数
pytest tests/test_arm.py -k jacobian -v                           # -k 子串等价于方式 A
```

### 1.2 全局测试（提交前 / 合并前 → 用它）

```bash
# 仓库根目录：
cd /home/dzxu/RoboTwinTutorial
python -m pytest projects/robotwin/tests -v      # 预期：35 passed
# 或在 projects/robotwin 下逐文件直跑（不依赖 pytest）：
cd projects/robotwin
for f in tests/test_*.py; do python3 "$f"; done
```

### 1.3 何时用哪个（速查）

| 场景 | 用法 |
|------|------|
| 改了 `arm.py`（被 tasks/policies 复用） | 先 `tests/test_arm.py`，再 `tests/test_tasks.py`、`tests/test_policies.py` |
| 改了 `dr.py`（被 tasks/benchmark/policies 复用） | 先 `tests/test_dr.py`，再全局 |
| 改了 `tasks.py`（回合物理） | `tests/test_tasks.py` + `tests/test_policies.py`（成功率阈值对物理敏感） |
| 提交前 | 全局 pytest + VS Code Testing 侧栏全绿 |
| 怀疑两解释器不一致 | 各跑一遍全局 + `diff <(python3 demo.py) <(python demo.py)`（应无输出） |

### 1.4 常见启动失败

- `ModuleNotFoundError: No module named 'dr'` —— 工作目录不在
  `projects/robotwin`，或用绝对路径从别处执行（`sys.path[0]` 是脚本目录
  而非 cwd）。tests/ 下的文件自带 `sys.path.insert`，照抄即可。
- VS Code 里 import 标红线 —— 从仓库根打开窗口后 Reload（SETUP.md §5）。

---

## 2. VS Code 断点调试

`F5` → 顶部下拉选配置（RoboTwin 的 3 条配置定义于
[.vscode/launch.json](../../.vscode/launch.json)，见 §2.1 注册说明）：

| 配置（注册名建议） | 用途 | 用法 |
|------|------|------|
| `RoboTwin: 核心入口（demo.py）` | 调试模式跑冒烟演示，`cwd` = `projects/robotwin` | 断点打在任意被调代码处即停 |
| `RoboTwin: 全局测试（pytest 全套件）` | 调试模式跑整套 pytest，`cwd` = 仓库根 | 断点打在任意被测代码处 |
| `RoboTwin: 单点测试（当前文件 + 测试名过滤）` | 调试**当前打开的** tests/test_*.py | F5 后弹输入框填测试名子串（如 `dose`、`gate`；留空 = 全部） |

`justMyCode` 默认 `true`（只在项目代码内停）；要单步进 numpy 内部（如
`np.linalg.lstsq`）时改为 `false`。终端里等价的直跑调试：
`python -m debugpy --wait-for-client tests/test_policies.py dose`。

### 2.1 launch.json 注册参数（供维护者照抄）

- 核心入口：`"program": "demo.py"`，`"cwd": "${workspaceFolder}/projects/robotwin"`。
- 全局测试：`"module": "pytest"`，`"args": ["projects/robotwin/tests", "-v"]`，
  `"cwd": "${workspaceFolder}"`。
- 单点测试：`"program": "${file}"`，`"args": ["${input:testFilter"]` + inputs
  定义 `"testFilter"`，`"cwd": "${workspaceFolder}/projects/robotwin"`。

### 推荐断点位置（函数级）

| 断点位置 | 在这里看什么 |
|----------|--------------|
| `arm.py::ik_both`（`cos_q2` 判越界一行） | 目标是否超出可达工作空间（变量表 A-1）；双解的 ±对称 |
| `arm.py::link_ik`（限位检查两行） | 解被限位拒绝的情况（变量表 A-2）——策略冻结的头号来源 |
| `tasks.py::BaseBimanualEnv.step`（`q_eff = ...` 一行） | 动作缩放 `action_scale` 的实际放大/缩小、限速裁剪（T-1/T-2） |
| `tasks.py::PushTask._task_dynamics`（`normal = ...` 一行） | 接触法向 `n̂`、推入量 `⟨Δee, n̂⟩`、圆盘位移（T-4/T-5） |
| `tasks.py::PickPlaceTask._task_dynamics`（吸附分支） | `holding` 翻转、吸附偏移 `offset`（T-6） |
| `policies.py::ScriptedPolicy._act_push`（stall 检测块） | `_best_dist`/`_since_improve` 与退回 back 的触发（P-3/P-4） |
| `policies.py::RobustPolicy._measure`（`lstsq` 一行） | 测量矩阵条件、`sol` 与真值连杆长的差（P-5/P-6） |
| `benchmark.py::run`（`seed = episode_seed(...)` 一行） | 种子派生四层索引（B-1） |

---

## 3. 重点观察变量表（核心章节）

约定：**形状**为断点处的 numpy 形状；**健康值**为基准场景（测试定种子）
实测；**异常信号**出现即有 bug 或配置错误。分组编号 A(rm)/T(asks)/
D(R 采样)/P(olicies)/B(enchmark)/M(etrics)。

### A：arm（断点 `arm.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| A-1 | `ik_both` | `cos_q2` | 目标半径的归一化余弦（标量） | \|cos_q2\| ≤ 1；可达网格回代误差 **1.67e-16**（`test_arm.py` 实测） | \|cos_q2\| > 1 → 目标不可达（返回双 None 是**正确行为**，看调用方怎么处理） |
| A-2 | `link_ik` | `sol` | 肘向解 (2,)，单位 rad | elbow=−1 时 q2∈[−2.9,−0.15] 全部可解 | 大量 None → 目标太靠近肩部内圈（r ≲ 0.21 m）或 q1 越界；策略会原地保持（冻结） |
| A-3 | `numerical_jacobian` vs `link_jacobian` | 两 J 之差 | (2,2) | **3.5e-11**（实测，门限 1e-6） | > 1e-6 → 解析式求导错 |

### T：tasks（断点 `tasks.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| T-1 | `BaseBimanualEnv.step` | `q_target` | 关节目标 (4,)，rad | 已裁到 Q1_LIMITS(−1.2,2.4)/Q2_LIMITS(−2.9,−0.15) | 出现 NaN/Inf → 上游动作坏 |
| T-2 | 同上 | `q_eff − q` | 缩放后增量 (4,) | 每步被限速裁到 ±0.15 rad；`action_scale`≠1 时增量同比例缩放 | 长期贴 0 → 目标=当前（策略冻结） |
| T-3 | `PushTask._task_dynamics` | `info["contact"]` | 接触布尔（bool） | push 阶段贴住圆盘时应持续 True | 推进中频繁 True/False 交替 → 末端在圆盘边缘打滑（几何失配的征兆） |
| T-4 | 同上 | `normal` | 接触法向 n̂ (2,)，无量纲 | 推土机跟随下 ≈ 推方向（点积 > 0.95） | n̂ 与推方向点积 < 0 → 末端越过圆心（反向推） |
| T-5 | 同上 | 圆盘位移/步 | (2,)，m | f=1 时 = 末端法向推入量（**实测单步 0.0200 m**，`test_tasks.py` 打印）；f=0.5 时减半（**0.0100 m**，ratio 0.500） | f<1 时推进总距离卡在上限 ≈ 初始间隙×f/(1−f)——这是物理，不是 bug（重推解之，见 P-4） |
| T-6 | `PickPlaceTask._task_dynamics` | `holding` / `self._hold_offset` | 吸附布尔 / 偏移 (2,)，m | 吸附后偏移逐位恒定（搬运中 ‖object−ee‖ 不变） | 搬运中偏移漂移 → 持握臂索引错；拎着到目标 `info["success"]` 仍 False 是**设计**（松开才算放置） |
| T-7 | `reset` | 场景采样 | goal/object (2,)，m | seen 目标 x∈[0.34,0.54]、unseen x∈[0.55,0.63]；都在最短 DR 臂（0.68 m）可达圈内 | 目标出现在圈外 → `SCENE_RANGES` 被改坏（IK 全 None，策略冻结） |

### D：dr（断点 `dr.py::sample`）

| # | 断点 / 来源 | 变量 | 含义 | 健康值 | 异常信号 |
|---|-------------|------|------|--------|----------|
| D-1 | `sample` | `s_left / s_right` | 左/右臂缩放系数（标量） | none=1.0；mild∈[0.95,1.05]；strong∈[0.85,1.15] | 超出档位区间 → `REGIME_BOUNDS` 或 rng 被误用 |
| D-2 | `sample` | `friction / noise / action` | 其余三维采样值 | mild: f∈[0.85,1]、noise∈[0,0.006]、act∈[0.95,1.05]；strong: f∈[0.7,1]、noise∈[0,0.015]、act∈[0.85,1.15] | `none` 档采出非名义值 → 区间退化被破坏 |
| D-3 | `WorldParams.__post_init__` | 构造校验 | — | 越界构造抛 ValueError（`tests/test_dr.py::test_invalid_params_rejected`） | 静默构造成功 → 校验被删 |

### P：policies（断点 `policies.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| P-1 | `_act_reach` | `self._plan` | 双臂关节目标 (4,) | none 档下规划后 15 步内收敛（实测 12/12 seed 成功） | `q_l/q_r` 为 None → 目标超**模型**工作空间（策略将原地保持到超时） |
| P-2 | `_act_push` | `direction` | 推方向 (2,)，单位矢量 | 指向目标（对象→目标连线） | NaN → 对象恰在目标上（`dist` 已 < 0.5·PUSH_TOL，应走保持分支） |
| P-3 | `_act_push` stall 块 | `_best_dist` / `_since_improve` | 历史最优距离 m / 未改善步数 int | 推进中持续刷新；重推后 `_best_dist` 重置 | `_since_improve` 长期 < 10 且 `_best_dist` 不动 → 钳位拖住（低摩擦的正常现象，等重推） |
| P-4 | `_act_push` | `self._phase` | 状态机 str | `""→side→back→push`；重推时 `push→back→push` | 卡在 `side` → 航点 IK 为 None 或永远到不了 0.02 邻域（看 A-2 与 T-2） |
| P-5 | `RobustPolicy._measure` | `sol` | 测量连杆长 (4,)，m | worst 误差 **0.0136 m**（8 回合 strong 实测；`test_policies.py` 打印） | > 0.06 → 探测读数坏（观测噪声超档位区间或 PROBE_Q 被改） |
| P-6 | 同上 | `measured_radius` | 测量半径（标量，m） | 与真值差 < 0.02（实测 ~0.005） | 系统性等于名义值 0.05 → `radius` 读数没进均值 |
| P-7 | 两策略对照（`test_policies.py` 打印） | 各档宏平均 | 标量 | calibrated **1.00→0.85→0.34**；robust **0.98→0.92→0.82**（N=40, seed0=0） | robust 的 strong 也塌到 < 0.5 → 观测噪声/测量流程坏；calibrated 的 none < 0.95 → 标定世界被污染（dr 的 none 档坏了） |

### B：benchmark / M：metrics

| # | 断点 / 来源 | 变量 | 含义 | 健康值 | 异常信号 |
|---|-------------|------|------|--------|----------|
| B-1 | `run` 内 `seed = episode_seed(...)` | 四层索引 | ints | demo 全网格下 strong 的 `regime_idx=2`；只传 `("strong",)` 时为 0（两种都合法但样本不同，见 §4 案例 4） | 同配置两次 run 记录不一致 → 存在未种子化的随机源 |
| B-2 | `run` 返回 | `records` 长度 | int | = 档位×任务×回合数（demo: 3×3×40=360） | 不符 → seen/unseen 对半分配坏 |
| M-1 | `metrics.success_rate` | 返回值 | 标量 ∈[0,1] | 空输入抛 ValueError（有测试） | 静默返回 NaN → 早失败被删 |
| M-2 | `macro_mean_and_worst` | `worst_name` | str | 全档位下最差任务恒为 **push**（demo 实测） | worst 变成 reach/pick_place → 某个任务物理被改坏 |
| M-3 | `generalization_gap` | 返回值 | 标量 ∈[−1,1] | calibrated: none=**0.00**、strong=**+0.25**；robust ≤ +0.23（demo 实测） | 单档 gap<0 属采样噪声（mild 实测 −0.03）；**持续** < 0 → seen/unseen 分层采样坏 |

---

## 4. 常见调试场景（症状 → 断点 → 看什么 → 结论）

### 案例 1：策略全程不动（episode 跑满 64 步、对象/末端原地）

- **症状**：某任务成功率归零；`info["t"]` 恒到 64；末端位置不变。
- **断点**：`policies.py` 各技能的 `if q is None: return idle` 分支。
- **看什么**：A-2 的 `sol` 是否被限位拒绝——head 号原因是目标点进了肩部
  内圈（r ≲ 0.21 m，elbow=−1 的 q2 会越过 −2.9 rad）。
- **结论**：检查场景范围（T-7）是否被改到太靠近肩部，或 Q2_LIMITS 是否被
  收紧（历史教训：−2.6 时拖曳式推物的 IK 在内圈无解，为此放宽到 −2.9，
  见 `arm.py` 的注释）。

### 案例 2：push 推到一半圆盘不走了（低摩擦档）

- **症状**：strong 档 push 成功率低；断点处圆盘位置不动、`contact=True`。
- **断点**：`policies.py::ScriptedPolicy._act_push` 的 stall 检测块（P-3/P-4）。
- **看什么**：`_since_improve` 是否涨过 10——低摩擦（f 低至 0.7）下末端被
  "圆心后 0.2R" 钳位拖住，位移传递物理的推送距离有硬上限
  （≈ 初始间隙 × f/(1−f)）。
- **结论**：这是物理不是 bug——重推机制（`_phase = "back"`）就是对策；
  若重推后仍卡死，查 `_phase` 是否意外停在 `side`（案例 1 的冻结）。

### 案例 3：pick_place "拎着也算成功"

- **症状**：修改状态机后 `info["success"]` 在 `holding=True` 时变 True。
- **断点**：`tasks.py::PickPlaceTask._success_gate`。
- **看什么**：`BaseBimanualEnv.step` 的成功判定必须**与**门条件相与
  （`dist < tol and self._success_gate()`）。
- **结论**：门条件是实现"松开后才算放置"（协议语义，教程 §5.6 的
  rollout 判定）的唯一位置；删掉它 calibrated 的 strong 档 pick_place 会
  虚高（实测修复前 0.95 → 修复后 0.53）。

### 案例 4：自己跑的数字和文档/demo 对不上

- **症状**：`bm.run(policy, regimes=("strong",))` 的成功率与 demo 表不同。
- **断点**：`benchmark.py::run` 的 `enumerate(regimes)` 与 `episode_seed`（B-1）。
- **看什么**：`regime_idx`——种子规则用的是**本次调用传入序列**中的下标：
  全网格下 strong=2，单档调用下 strong=0 → 两种调用采到不同世界参数样本
  （都确定、都合法）。
- **结论**：对表时用与文档相同的调用方式（demo = 全网格 + seed0=0 +
  episodes_per_task=40）；要复现某条记录，用记录里的 `seed` 直接
  `env.reset(seed=rec.seed, split=rec.split)`。

---

## 5. 测试怎么写

新测试放在 `tests/test_<模块名>.py`，遵循本套件既有惯例：

1. **命名 `test_<行为>`**——名字说清"验证什么行为"，因为单点过滤按子串
   匹配（§1.1），名字是过滤键。范例：`test_calibrated_dose_monotone_decrease`、
   `test_push_displacement_direction_and_friction`。注意 Python 标识符不能
   含 `-`（`1e-8` 要写成 `1e_8`）。
2. **deterministic seed**——一切随机性走 `np.random.default_rng(seed)`（场景、
   DR 参数、探测各给独立种子），使打印的数字与断言阈值可精确复现；禁止
   `np.random.seed`/全局随机态。
3. **断言阈值旁注明理由与实测值**——照套件惯例（"门限理由：实测 X，留 Y
   裕量"）写清期望值的量级来源；成功率类阈值在实测值上留 ≥0.1 裕量
   （跨 numpy 版本的边界翻转保护）。
4. **文件尾部带单点过滤主入口**——照抄 slam 的 `__main__` 块（匹配 `test_`
   前缀 + 子串过滤；无匹配列出全部测试名并 exit 1；逐个运行打印
   PASS/FAIL 与回溯；末行 `N/M tests passed.`）。
5. **运行时长预算 < 15 s / 文件**——当前最慢的是 `tests/test_policies.py`
   （~3.0 s，含 3 次 N=40 的全网格评测）；Monte-Carlo 类测试通过减少回合数
   而非删断言来控制时长。指标断言从 `metrics.py` 取实现，勿在测试里重写公式。
6. **写完自查**：`python tests/test_<新文件>.py` 全绿 + 该模块断点处的观察量
   （§3 变量表）落在健康值内 + 若新增了被监视变量，同步更新本文件的变量表
   （CONSTRAINTS §4.5 质量要求）。
