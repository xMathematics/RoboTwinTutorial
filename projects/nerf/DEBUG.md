# NeRF 调试与测试教程（DEBUG.md）

> **适用对象**：需要改动 / 排查本项目代码的开发者。规范依据
> [CONSTRAINTS.md §4.5](../../CONSTRAINTS.md)；VS Code 配置总览见
> [.vscode/SETUP.md](../../.vscode/SETUP.md)。所有命令、变量名、行号均对照当前代码核实，
> 健康值引自 [METRICS.md](METRICS.md) 的实测基线。

---

## 1. 环境与两种测试

### 1.1 解释器

- **必须用 conda 的 `llm_env`**（本项目依赖 torch；numpy/pytest 也在其中）：
  终端 `conda activate llm_env`，或单条命令用 `conda run -n llm_env python ...`。
- VS Code 已由 `.vscode/settings.json` 把默认解释器指向 llm_env；若终端报
  `No module named torch`，是解释器选错，见 [TUTORIAL.md §7 Q2](TUTORIAL.md)。
- 版本参考（实跑环境）：torch 2.11.0+cu130、numpy 2.2.6、pytest 9.1.1。
  `tests/` 与 `run_nerf.py` 均自带 `sys.path.insert`，从仓库根或项目根都能跑。

### 1.2 单点测试（改了一个模块 → 只跑相关测试）

两种等价写法（`tests/test_core.py` 8 个测试、`tests/test_metrics.py` 7 个测试，共 15 个）：

```bash
# 直跑 + 子串过滤（不依赖 pytest；子串无匹配时会列出全部可用测试名）
conda run -n llm_env python tests/test_core.py volume_render
#   → PASS test_volume_render_empty_space_is_black
#     PASS test_volume_render_single_opaque_point
#     2/2 tests passed
conda run -n llm_env python tests/test_metrics.py ssim        # → 3/3 tests passed

# pytest 单点（::后跟完整测试名）
conda run -n llm_env python -m pytest tests/test_core.py::test_model_forward_shapes -v
#   → 1 passed in 0.98s
```

**何时用**：改了 `render.py` → `python tests/test_core.py volume_render`；
改了 `metrics.py` → `python tests/test_metrics.py`；改了 `sampling.py` →
`python tests/test_core.py sample`（过滤出 `test_stratified_sample_in_bins` 与
`test_hierarchical_sample_count_and_range`）。

### 1.3 全局测试（提交前 / 合并前跑全套）

```bash
cd projects/nerf
conda run -n llm_env python -m pytest tests/ -v
#   → 15 passed in 1.94s（实跑）
```

也可从仓库根跑：`conda run -n llm_env python -m pytest projects/nerf/tests -q`（实测同样
15 passed）。无 pytest 时可逐文件直跑：`python tests/test_core.py`、`python tests/test_metrics.py`
（不带参数 = 全部测试）。

| 场景 | 用哪种 |
|------|--------|
| 改了单个模块 / 单个函数 | 单点（子串过滤，秒级反馈） |
| 改动跨模块、或准备提交 | 全局（15 个测试 ~2 秒） |
| 怀疑训练管线坏（非单元逻辑） | 跑 §2 的 tiny 冒烟训练，看 final loss 与 [test] PSNR |

---

## 2. VS Code 断点调试

### 2.1 launch.json 的 4 条 NeRF 配置怎么选

`.vscode/launch.json`（工作区 = 仓库根）为 nerf 提供了 4 条配置，均为 debugpy 启动
`run_nerf.py`、`cwd` 固定 `projects/nerf`、`justMyCode: true`：

| 配置名 | 等价命令 | 什么时候用 |
|--------|---------|-----------|
| **NeRF: 训练 (tiny 冒烟)** | `--config data/lego --mode train --exp lego_smoke --tiny` | **首选调试入口**：分钟级进入训练循环，每步都能在断点停下。注意其默认参数指向 `data/lego`——没有官方数据时，把 args 里的 `--config` 改成 `data/demo_scene`（并用 §2.2 的命令先生成）即可立刻跑通 |
| **NeRF: 训练 (全量)** | `--config data/lego --mode train --exp lego_full --steps 200000` | 观察长程行为（lr 衰减末端、收敛平台期）；调试时请配合 `--steps` 调小 |
| **NeRF: 测试 (PSNR/SSIM)** | `--config data/lego --mode test --exp lego_full --ckpt logs/lego_full/latest.pt` | 调试评测链路：render_image → psnr/ssim → metrics.json |
| **NeRF: 渲染相机轨迹** | `--config data/lego --mode render --exp lego_full --ckpt ... --frames 120` | 调试 `look_at / render_path_poses / get_rays(torch 版)` |

另有通用配置 **Python: 当前文件**（调试当前打开的任何脚本，如 `tests/test_core.py`）。
不想记命令行参数时，`Terminal → Run Task` 里有对应的任务条目（`NeRF: 运行核心测试`、
`NeRF: 训练 (tiny 冒烟)` 等，见 `.vscode/tasks.json`）。

### 2.2 冒烟训练调试的标准流程

1. 终端生成演示数据（一次性）：`conda run -n llm_env python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32`；
2. launch.json 选 **NeRF: 训练 (tiny 冒烟)**，把 `--config` 改为 `data/demo_scene`；
3. 按下表打断点 → F5 → 每次命中在调试控制台查变量（变量表见 §3）；
4. tiny 档 batch 256、N_c=32、N_f=64，变量都很小，单步飞快。

### 2.3 推荐断点（函数级，行号已对照当前代码）

| 断点位置 | 停下时能看到什么 |
|---------|-----------------|
| `nerf/render.py::volume_render` 第 **40** 行（`alpha = 1.0 - torch.exp(...)`） | 体积渲染的入口三件套：入参 `sigma`、`deltas` 与刚算出的 `alpha`——判断"渲染为什么黑/为什么糊"的第一现场 |
| `nerf/render.py::volume_render` 第 **48** 行（`weights = transmittance * alpha`） | 合成权重 `weights`（Eq.5）——层次采样的分布源头 |
| `nerf/sampling.py::hierarchical_sample` 第 **65** 行（`idx = torch.searchsorted(cdf, u)`） | CDF 采样的核心行：`cdf`、随机数 `u`、查表结果 `idx`、以及其后的 `lo/hi` |
| `nerf/model.py::NeRF.forward` 第 **97** 行（`sigma = F.relu(self.sigma_head(h))`） | 网络两个输出头之一：密度 σ ≥ 0；第 105 行 `rgb = torch.sigmoid(...)` 则看颜色 |
| `nerf/trainer.py::train_nerf` 第 **114** 行（`loss = F.mse_loss(...) + ...`） | 每步的 batch（`batch_ro/batch_rd/batch_target`）与渲染输出 `out`；第 **119** 行 `loss.backward()` 是反传现场，第 **88** 行看 `lr` 衰减 |
| `run_nerf.py` test 模式第 **178**–**183** 行（`pred = render_image(...)` … `psnrs.append(psnr(pred_np, gt_np))`） | 逐张评测现场：`pred_np` 与 `gt_np` 并排对比，直接在调试控制台算 `psnr` |

### 2.4 justMyCode 何时关闭

所有配置默认 `"justMyCode": true`——只在项目代码内停，栈里 numpy/torch 的帧被折叠。
需要单步**进入 torch 内部**查证行为时（例如想确认 `torch.searchsorted` 对边界值的处理、
`torch.cumprod` 的数值下溢），把该配置改为 `"justMyCode": false`，然后可以在
调试栈里点进 torch 源码帧。调试完记得改回来，否则每次单步都会误入库代码。

---

## 3. 重点观察变量表（核心章节）

> 每个变量名/形状均 grep 当前代码核实；"健康值"为 demo_scene 冒烟基线（CUDA 实测）。
> R=batch 光线数，N_c/N_f=采样点数，B=batch_size。tiny 档：R=256、N_c=32、N_f=64。

| 断点位置 | 变量 | 含义 | 形状 | 健康值 | 异常信号 |
|---------|------|------|------|--------|---------|
| `render.py:40` volume_render | `sigma`（入参） | 网络预测密度 σ | [R, N_c, 1] | ≥ 0（ReLU 保证）；场景内部应有明显非零 | **恒 0 → 渲染全黑**（激活/输入问题，见 §4.1） |
| `render.py:40` | `deltas` | 相邻采样深度 δ_i，末位 1e10 | [R, N_c] | 前段 ≈ (far−near)/N_c（默认 64 点 ≈ 0.06；tiny 32 点 ≈ 0.125），末位 = 1e10 | 出现 NaN；末位丢失（eq.3 的 T 会错） |
| `render.py:40` | `alpha` | 不透明度 α_i = 1−e^(−σ_i·δ_i) | [R, N_c] | σ=0 → **全 0**；σ=1e4 → **≈1**（两个极端由 `test_volume_render_empty_space_is_black` / `test_volume_render_single_opaque_point` 固定） | 值域越出 [0,1)；σ 很大但 α 不大 → deltas 量纲错 |
| `render.py:48` | `weights` | 合成权重 w_i = T_i·α_i（Eq.5） | [R, N_c] | 非负、沿 N_c 求和 ≤ 1；场景被积满时 sum ≈ 0.1~1 | `weights.sum(-1)` 恒 0 → 网络没学到任何密度；> 1+ε → T 计算错 |
| `render.py:49` | `color` | 光线渲染颜色 C(r) | [R, 3] | ∈ [0, 1]（权重凸组合） | NaN / 越界 |
| `sampling.py:65` hierarchical_sample | `cdf` | 权重归一化后的累积分布（首 0 尾 1） | [R, N_c+1] | 单调不减、`cdf[...,-1] == 1.0`（全零权重有 +1e-5 保护） | NaN；末位 < 1（权重为 0 且保护失效） |
| `sampling.py:65` | `idx`（及 `u`） | 逆变换采样查表结果（u ∈ [0,1)） | idx: [R, N_f] 整数；u: [R, N_f] | idx ∈ [1, N_c−1]（越界会被 clamp 掩盖，检查前先看 clamp 行） | idx 全 0 或全 N_c−1 → 权重塌缩到一个点（训练早期属正常，持续如此才异常） |
| `model.py:97` NeRF.forward | `sigma` | 密度头输出（ReLU 后） | [R, N_c, 1]（按采样点调用时） | ≥ 0；训练早期接近 0，之后出现 O(1)~O(10) 的大值 | 恒 0 → 上游编码/输入错（§4.1）；爆炸到 1e4+ → loss NaN 前兆 |
| `model.py:105` | `rgb` | 颜色头输出（sigmoid 后） | [R, N_c, 3] | ∈ [0, 1]；未训练时约 0.5 附近（实测 0.435~0.548，Xavier 初始化） | 大面积饱和 0/1 → 检查 lr 或数据范围 |
| `trainer.py:88` train_nerf | `lr` | 当前学习率（每步重设） | 标量 | 5e-4 起按 `lr_decay^(step/steps)` 指数衰减至 5e-5（80 步末步实测 ≈ 5.1e-5；tqdm 角标显示的是第 0 步的 5.00e-04） | 不衰减 → `cfg.lr_decay` 被改；NaN → lr 过大或数据有 NaN |
| `trainer.py:114` | `loss` | Eq.7：coarse 与 fine 两项 MSE 之和 | 标量 | 80 步冒烟 **0.2024 → 0.0247**（final）；真实数据全量训练应持续下降 | 卡在 ~0.1 不降（§4.2）；NaN |
| `trainer.py:94–96` | `batch_ro / batch_rd / batch_target` | 抽样的光线原点/方向/真值颜色 | 各 [B, 3] | target ∈ [0, 1]；rd 为像素方向（未归一化） | target 出界 → 读图/白底合成错；三者行数不一致 → 光线与图像错位 |
| `run_nerf.py:178–183` test | `pred_np` / `gt_np` | 渲染图 / 真值图 | 各 [H, W, 3] | 同形状、同 [0,1]；**PSNR 16.23**（testskip=8 评 1 张）/ **19.23**（8 张全评） | **PSNR < 10 → 数据或位姿错**（METRICS.md §1.3）；pred 全黑 → 回到 volume_render 查 σ |
| `metrics.py:91` ssim | `ssim_map` | 逐像素 SSIM 图 | [H, W, 3] | 均值 **0.7682**（1 张）/ **0.8346**（8 张） | ≈ 0 或负 → 渲染与真值结构不相关（位姿/场景对不上，比 PSNR 更早暴露） |
| `make_demo_data.py:121` main | `frames`（写入 `transforms_*.json`） | 帧元数据列表 | 长度 = n_train/n_test | 每项含 `file_path`（`./train/r_000`）与 4×4 `transform_matrix`（末列平移 ≈ 半径 4 处） | transform_matrix 末列全 0 → 相机全在原点，多视角退化 |

> 补充：`render_rays` 的 `t_coarse`（[R, N_c]，∈ [near, far] = [2, 6]）与 `t_all`
> （[R, N_c+N_f]，合并后已 sort）也是快速判断采样是否越界的好变量。

---

## 4. 常见调试场景

### 4.1 症状：渲染全黑（训练正常但 test 模式 / render 模式输出黑图）

- **断点**：`render.py:40`（volume_render 入口）。
- **看什么**：`sigma` 是否恒 0？若 σ 有值，再看 `alpha`、`weights.sum(-1)`。
- **结论路径**：
  - σ 恒 0 → 激活函数/输入问题：σ 由 `sigma_head → ReLU` 产生（model.py:97），
    输入侧先过位置编码（encoding.py）；维度错配（如把未编码坐标喂给 `in_dim=60` 的
    网络）会直接抛 shape 错误，所以 σ 恒 0 更可能来自**输入量纲/尺度**——先确认
    `Config` 的 `l_xyz=10, l_dir=4` 没被改、坐标经 `positional_encoding` 后量级正常。
  - σ 有值但 weights ≈ 0 → 场景不在 [near, far] = [2, 6] 内：demo/lego 的相机半径 4、
    物体在原点边长 2 立方体内；若换了自备数据（相机距离 10+），必须同步改 `Config.near/far`，
    否则采样点全部落在真空，α 全 0 → 黑图。

### 4.2 症状：loss 不降（停在 ~0.1 量级震荡）

- **断点**：`trainer.py:88`（lr 重设）与 `trainer.py:114`（loss 计算）。
- **看什么**：`lr` 是否从 5e-4 起且按步衰减；`batch_target` 是否 ∈ [0,1]；
  `out["rgb_coarse"]` 与 `batch_target` 是否"同池同序"（三者共用同一个 `idx`
  `torch.randint` 抽样，trainer.py:93——若有人改代码让光线与图像用两个独立随机数，
  光线-颜色错位，loss 永远降不下去）。
- **结论**：target 正常但 loss 不降 → 检查 `perturb=True` 是否被误关（关掉会削弱正则）、
  `lr_decay` 是否被改成 1（lr 恒 5e-4，后期震荡不收敛）。另外注意 tqdm 角标每 100 步
  才刷新（80 步冒烟里它显示第 0 步的 0.2024），真实进度看 `final loss` 或在 114 行断点
  逐步观察。

### 4.3 症状：test 模式 PSNR 异常低（< 10 dB，甚至 0 附近）

- **断点**：`run_nerf.py:178–183`（逐张评测现场）。
- **看什么**：调试控制台里并排比较 `pred_np.mean()` 与 `gt_np.mean()`（差 > 0.3 说明
  整体亮度/内容对不上）；`rd`（get_rays 产物）与 `focal` 是否 half_res 一致
  （图像减半但 focal 没减半 → 视场角翻倍 → 内容错位）；`cfg.near/far` 是否罩住场景。
- **结论**：单张极差、其余正常 → 看 `metrics.json` 的 `psnr_list` 找最差视角，多半是
  该位姿离训练分布远（正常现象）；**全部**极差 → 位姿/内参解读错误（c2w 的约定是
  "相机朝 −z 看"，`rays.py` 的 `dirs` 第三分量是 −1）或评估时把真值当预测
  （PSNR 恰 100 dB / SSIM 恰 1.0 是另一个极端信号，见 METRICS.md §1.3）。

---

## 5. 测试怎么写

1. **文件与命名**：放 `tests/test_<模块主题>.py`；函数名 `test_<行为>`
   （如 `test_volume_render_empty_space_is_black`——名字直接说明被固定的行为）；
   文件头部 docstring 用中文写覆盖范围与跑法。
2. **确定性 seed**：随机输入必须固定种子——torch 侧 `torch.manual_seed(0)`、
   numpy 侧 `rng = np.random.default_rng(0)`（参照 `test_core.py::test_end_to_end_forward_backward`
   与 `test_metrics.py` 全部测试），保证重跑逐位一致。
3. **优先解析可手算的边界情形**：像"σ 全 0 → 黑"、"σ=1e4 → 该点颜色胜出"、
   "已知 MSE=0.01 → 恰 20 dB"这样能独立手算的断言，比随机数对随机数更有诊断力；
   需要交叉验证时写独立的朴素实现（参照 `test_metrics.py` 的 `_naive_*` 系列）。
4. **单点过滤主入口模板**（照抄 `tests/test_core.py` 的 `__main__` 块，两个测试文件
   完全一致；直接执行时支持子串过滤，无匹配列出可用测试名并退出码 1）：

   ```python
   if __name__ == "__main__":
       import traceback

       # 直接执行时：收集所有 test_ 开头的函数逐个运行，汇总通过率（不依赖 pytest）
       # 单点测试：python tests/test_core.py <测试名子串>；不带参数 = 全部测试。
       pattern = sys.argv[1] if len(sys.argv) > 1 else ""
       fns = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and pattern in k]
       if not fns:
           print(f"没有匹配 '{pattern}' 的测试；可用测试：")
           for k in sorted(globals()):
               if k.startswith("test_"):
                   print("  ", k)
           sys.exit(1)
       failed = 0
       for fn in fns:
           try:
               fn()
               print(f"PASS  {fn.__name__}")
           except Exception:
               failed += 1
               print(f"FAIL  {fn.__name__}")
               traceback.print_exc()
       print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
       sys.exit(1 if failed else 0)
   ```

5. **时长预算**：每个单元测试目标 **< 1 秒**（当前 15 个全套实测 1.94 秒）；涉及
   `render_rays` 端到端的测试把 `n_coarse/n_fine` 调小（参照 `test_end_to_end_forward_backward`
   用 8/16 点）。**不要**把 80 步冒烟训练（数秒）放进 pytest 套件——它属于手动冒烟，
   走 `run_nerf.py --tiny`（见 [TUTORIAL.md §2.2](TUTORIAL.md)）。
