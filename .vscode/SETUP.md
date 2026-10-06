# VS Code 配置教程（本项目开发与测试的唯一入口）

> 本项目**完全使用 VS Code 开发与测试**。本教程解释 `.vscode/` 四个配置文件各自管什么、
> Anaconda 环境如何被 VS Code 使用、以及"运行 / 测试 / 调试"三种日常操作的标准做法。
> 相关规范：[CONSTRAINTS.md §4.5](../CONSTRAINTS.md)（DEBUG.md）、§4.6（Anaconda 清单）、§4.8（本节）。

---

## 0. 前置条件（一次性）

1. 安装 **Anaconda** 并确认环境清单已创建：仓库根目录执行
   `conda env create -f environment.yml && conda activate llm_env`
2. 用 VS Code 打开**仓库根目录**（`File → Open Folder → RoboTwinTutorial`）——
   所有配置都按"工作区=仓库根"写好，从子目录打开会失效。
3. 首次打开会提示"是否安装推荐扩展"（来自 `.vscode/extensions.json`），选**安装**：
   `ms-python.python`（Python 支持）、`ms-python.vscode-pylance`（静态分析）、
   `ms-python.debugpy`（调试器）、`ms-python.black-formatter`（格式化）、
   `ms-toolsai.jupyter`（笔记本支持，教程中零星用到）。

## 1. 四个配置文件各管什么

| 文件 | 管什么 | 关键内容 |
|------|--------|---------|
| `settings.json` | **Anaconda 环境写入 VS Code**：默认解释器指向 conda `llm_env`、终端自动激活、pytest 测试发现范围、代码智能分析的搜索路径、保存时自动格式化 | `python.defaultInterpreterPath` = `/home/dzxu/anaconda3/envs/llm_env/bin/python` |
| `launch.json` | **调试**：21 条调试配置——NeRF 6 条（tiny 冒烟训练/全局测试/单点测试/全量训练/测试/渲染）、SLAM 5 条（核心入口 demo.py/全局测试/单点测试/两个代表性单点示例）、3D 重建 3 条、控制规划 3 条、RoboTwin 3 条（各为核心入口/全局测试/单点测试）、通用"当前文件"1 条 | 见 §4 |
| `tasks.json` | **任务**：18 条一键任务（五项目各有核心入口冒烟、全局测试、单点过滤测试；NeRF 另有全量训练/测试/渲染），`Terminal → Run Task` 调用 | 测试任务分组为 `test` |
| `extensions.json` | 推荐扩展清单 | 打开仓库时自动提示 |

**新增项目时的统一动作**：`settings.json` 的 `pytestArgs` / `extraPaths` / `PYTHONPATH`
各加一行该项目的路径；`tasks.json` / `launch.json` 加该项目的全局测试与单点测试条目
（照抄 SLAM 三条改路径即可）。规范见 [CONSTRAINTS.md §4.8](../CONSTRAINTS.md)。

## 2. 解释器与环境（Anaconda 如何进入 VS Code）

- **默认解释器已写死**在 `settings.json`：`python.defaultInterpreterPath` 指向
  conda 的 `llm_env`。命令面板（`Ctrl+Shift+P`）→ `Python: Select Interpreter`
  应显示 `llm_env (/home/dzxu/anaconda3/envs/llm_env/bin/python)` 且带 ✓。
- **新开终端自动激活** llm_env（`python.terminal.activateEnvironment: true`）——
  终端提示符前出现 `(llm_env)` 即正常；若没有，手动 `conda activate llm_env`。
- **五个项目的环境说明**：
  - `projects/nerf`（torch）：**必须**用 llm_env；
  - `projects/slam`、`projects/3d_reconstruction`、`projects/control_planning`、
    `projects/robotwin`（纯 numpy）：llm_env 与系统 python3（numpy≥1.26）均可，
    统一建议用 llm_env（全套件已在两种环境验证通过，262 项测试）。
- 验证：终端里 `python -c "import torch, numpy, pytest; print(torch.__version__, numpy.__version__, pytest.__version__)"`。

## 3. 运行测试（三种方式，单点 vs 全局）

| 方式 | 操作 | 适用 |
|------|------|------|
| **Testing 侧栏**（推荐日常） | 左侧烧杯图标 → pytest 自动发现 `projects/{nerf,slam}/tests` → 点单个测试旁的 ▷ = **单点测试**；点文件/目录级 ▷ = 全局 | 改一个模块后快速验证；提交前跑全套 |
| **任务** | `Terminal → Run Task` → `NeRF: 全局测试（pytest 全套件 15 项）` / `SLAM: 全局测试（pytest 全套件 80 项）` / `NeRF|SLAM: 单点测试（当前文件 + 过滤）`（会提示输入测试名子串，留空=全部） | 不想记命令 |
| **终端命令** | 单点：`python tests/test_fastslam.py gate`（子串过滤，无匹配会列出可用测试名）；全局：`python -m pytest projects/slam/tests -v` 或逐文件直跑 | 远程/脚本场景 |

**单点 vs 全局怎么选**：改了某个模块 → 先跑该模块的测试文件（或单点）；
提交前 / 合并前 → 全局（nerf 与 slam 各自 `tests/` 全部）。
详见各项目 `DEBUG.md`。

## 4. 调试（launch.json 用法）

- 左侧"运行和调试"（`Ctrl+Shift+D`）→ 顶部下拉选配置 → `F5` 启动。
- **NeRF / SLAM: 单点测试（当前文件 + 测试名过滤）**：打开任意 `tests/test_*.py` 后选它，
  F5 时会弹出输入框填测试名子串（如 `hartley`），在断点处停下后即可检查变量。
- **NeRF: 训练 (tiny 冒烟)** 与 **SLAM: 核心入口（demo.py）** 开箱即跑（前者先用
  `scripts/make_demo_data.py` 生成 `data/demo_scene`，后者无需数据）；
  标注"需 lego"的 NeRF 配置要先下载 Blender lego 数据集（见 projects/nerf/DEBUG.md）。
- 断点建议与**重点观察变量表**（每个模块调试时看什么、健康值是什么）：
  见 [projects/slam/DEBUG.md](../projects/slam/DEBUG.md) 与 [projects/nerf/DEBUG.md](../projects/nerf/DEBUG.md)。
- `justMyCode` 默认 `true`（只在项目代码内停）；要单步进入 numpy/torch 内部时改为 `false`。

## 5. 常见问题

| 症状 | 原因与修复 |
|------|-----------|
| Testing 侧栏发现不到测试 | 解释器不是 llm_env（§2 验证命令）；或 VS Code 从子目录打开（必须开仓库根） |
| 运行 `run_nerf.py` 报 `No module named torch` | 解释器选错。`Python: Select Interpreter` 选 llm_env |
| 终端没有 `(llm_env)` 前缀 | `settings.json` 的自动激活被关了；手动 `conda activate llm_env`，或重启终端 |
| 单点测试提示"没有匹配的测试" | 子串写错——输出会列出全部可用测试名，照着选 |
| NeRF 调试配置报找不到 `data/lego` | lego 是 Blender 官方数据，需先下载（见 projects/nerf/DEBUG.md）；开箱即跑请用 tiny 冒烟配置（`data/demo_scene`，先跑 `scripts/make_demo_data.py`） |
| 合并收集时 `import metrics` 拿到别的项目的模块 | 五个项目各有同名 `metrics.py`（CONSTRAINTS §4.7）：pytest 侧用 `--import-mode=importlib`（settings.json 已配），代码侧各新项目的测试/benchmark 用"按路径加载器"独立加载（见各文件注释）。**不要在新代码里裸 `import metrics`**，照抄 loader 或用包路径（如 nerf 的 `nerf.metrics`） |
| `import` 项目内包标红线 | VS Code 窗口需重载（`Developer: Reload Window`）让 `extraPaths` 生效 |
| 保存后代码被重排 | `black-formatter` 在保存时格式化（ruler=88）；属预期行为，不要手写超 88 列 |

## 6. 配置变更流程

改动 `.vscode/` 任何文件后：`Developer: Reload Window` 生效；涉及 §4.5/§4.8 规范的
结构变更（新增配置条目、更换解释器路径）须同步更新本教程与 CONSTRAINTS.md。
