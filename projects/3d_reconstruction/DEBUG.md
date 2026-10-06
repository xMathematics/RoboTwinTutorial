# 3D 重建教学代码库调试与测试教程（DEBUG）

> 零基础基准撰写。姊妹文档：[TUTORIAL.md](TUTORIAL.md)（代码导读）｜
> [METRICS.md](METRICS.md)（指标健康值——本文变量表的健康值与它保持一致）｜
> [.vscode/SETUP.md](../../.vscode/SETUP.md)（VS Code 配置总教程）。
> 规范出处：[CONSTRAINTS.md §4.5](../../CONSTRAINTS.md)。

---

## 1. 环境与两种测试

**解释器**：本项目纯 numpy，两套解释器均验证通过（55/55）——

| 解释器 | numpy | 全套件耗时（实测 `pytest tests/ -q`） |
|--------|-------|----------------------------------------|
| conda `llm_env`（`~/anaconda3/envs/llm_env/bin/python`，pytest 9.1.1） | 2.2.6 | 15.5 s |
| 系统 `python3`（pytest 7.x） | 1.26.4 | 16.6 s |

任选其一即可；VS Code 默认指向 `llm_env`（见 SETUP.md §2）。本项目没有 SLAM/nerf
那类 SVD/特征分解路径，测试阈值远大于版本间浮点差；demo 的指标在**打印末位**
（~2e-6 m²）内一致——BLAS 求和顺序的微小差异经最近邻并列选择放大属预期
（同一环境内重跑则逐位可复现）。

### 1.1 单点测试（改了一个模块 → 用它）

**方式 A：直跑过滤**（每个测试文件的 `__main__` 块都内置，推荐）：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/3d_reconstruction   # 必须在本目录（原因见 §1.4）

python tests/test_marching.py normals    # 只跑名字含 "normals" 的测试
# 预期输出：
# [mt] box: V=3870 T=7736 axis-aligned normals 0.789
# PASS test_box_extraction_extent_and_face_normals
# PASS test_normals_point_outward
# 2/2 tests passed.
python tests/test_scene.py sdf           # 前缀更宽：4 个名字含 sdf 的测试都命中
python tests/test_marching.py zzz        # 子串无任何匹配：列出全部可用测试名再退出（exit 1）
python tests/test_tsdf.py                # 不带参数 = 该文件全部测试（8 个，~0.2 s）
```

**方式 B：pytest 过滤**：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/3d_reconstruction
pytest tests/test_marching.py::test_normals_point_outward -v        # 精确到函数
pytest tests/test_tsdf.py -k zero -v                                # -k 子串等价于方式 A
```

### 1.2 全局测试（提交前 / 合并前 → 用它）

```bash
cd /home/dzxu/RoboTwinTutorial/projects/3d_reconstruction
python -m pytest tests/ -q                   # 预期：55 passed（~16 s，双环境）
# 或逐文件直跑（不依赖 pytest）：
for f in tests/test_*.py; do python3 "$f"; done
```

各测试文件实测耗时（系统 python3）：`test_scene.py` 0.17 s ｜ `test_tsdf.py` 0.23 s ｜
`test_marching.py` 1.3 s ｜ `test_metrics.py` 0.15 s ｜ `test_poisson.py` 5.2 s ｜
`test_plane_sweep.py` 9.8 s ｜ `test_splatting.py` 1.3 s。

### 1.3 何时用哪个（速查）

| 场景 | 用法 |
|------|------|
| 改了 `tsdf.py`（融合/查询） | 单点：`python tests/test_tsdf.py`；再跑 `tests/test_marching.py`（消费 tsdf 的场） |
| 改了 `scene.py`（被下游全部依赖） | 先 `tests/test_scene.py`，再全局 |
| 改了 16 case 表 / 提取核心 | `python tests/test_marching.py table`（穷举表自检）+ 全局 |
| 改了 `poisson.py`（涂抹/求解/等值面） | `python tests/test_poisson.py fft`（求解器解析对）→ `python tests/test_poisson.py`（球重建端到端） |
| 改了 `plane_sweep.py`（单应/NCC/取优） | `python tests/test_plane_sweep.py homography`（几何）→ 全文件（深度/置信度/融合） |
| 改了 `splatting.py`（投影/排序/合成） | `python tests/test_splatting.py analytic`（闭式对照）→ 全文件 |
| 提交前 | 全局 pytest + VS Code Testing 侧栏全绿 |
| 怀疑环境差异 | 两个解释器各跑一遍全局；同一环境内重跑应逐位一致，跨环境 demo 指标允许打印末位差（~2e-6 m²，BLAS 求和顺序 × 最近邻并列），差异超出该量级才说明用了未固化的随机性 |

### 1.4 常见启动失败

- `ModuleNotFoundError: No module named 'scene'` —— 工作目录不在
  `projects/3d_reconstruction`，或用 `python3 /path/to/script.py` 从别处执行
  （`sys.path[0]` 是脚本目录而非 cwd）。tests/ 下的文件自带 `sys.path.insert`，
  照抄即可。
- VS Code 里 import 标红线 —— 从仓库根打开窗口后 Reload（SETUP.md §5）。

---

## 2. VS Code 断点调试

`F5` → 顶部下拉选配置（3 条 3D 重建配置注册于
[.vscode/launch.json](../../.vscode/launch.json)，解释器均为 conda `llm_env`）：

| 配置名（与 launch.json 一致） | 用途 | 用法 |
|------|------|------|
| `3D 重建: 核心入口（demo.py）` | 调试主管线冒烟，`cwd` = projects/3d_reconstruction | 断点打在 demo.py 或任意被调模块处即停 |
| `3D 重建: 全局测试（pytest 全套件）` | 以调试模式跑 `pytest tests/` | 断点打在任意被测代码处即停 |
| `3D 重建: 单点测试（当前文件 + 测试名过滤）` | 调试**当前打开的** tests/test_*.py | F5 后弹输入框填测试名子串（如 `normals`、`zero`；留空 = 全部） |

`justMyCode` 默认 `true`（只在项目代码内停）；要单步进 numpy 内部（如
`np.gradient`）时在 launch.json 中改为 `false`。终端里等价的直跑调试：
`python -m debugpy --wait-for-client tests/test_marching.py normals`。

### 推荐断点位置（函数级，行号为 2026-10-06 代码定稿时的精确值）

| 断点位置 | 在这里看什么 |
|----------|--------------|
| `scene.py::render_depth`（L388 `p = t[active, None] * ...`） | 活跃光线的当前采样点、`s` 步长是否单调收缩（变量表 S-1/S-2） |
| `scene.py::render_depth`（L392 `depth[idx[hit]] = ...`） | 命中光线的深度写入、无回波（inf）比例（S-3） |
| `scene.py::make_orbit_poses`（`R_wc = np.column_stack(...)` 一行） | `right/down/forward` 三轴的正交右手性（det(R)=+1；历史 bug 见 §4 案例 2） |
| `scene.py::render_dataset`（L443 `sigma = ...`） | 逐像素噪声标准差 σ = c·z² 的量级（S-6） |
| `tsdf.py::TSDFVolume.integrate`（L146 `s = zpix - z`） | 全体体素的离散观测 (V,)，分布应集中在 ±0.1 m 内（T-1） |
| `tsdf.py::TSDFVolume.integrate`（L147 `band = ...`） | 带内掩码密度：updated 返回值 = band.sum()（T-2） |
| `tsdf.py::TSDFVolume.integrate`（L157–164 `w / f_obs / w_new / f_new`） | (4.2)/(4.3)/(4.8) 递推一步的全貌（T-3～T-5） |
| `tsdf.py::TSDFVolume.extraction_field`（L227 附近 `field[self.weight == 0.0] = np.nan`） | NaN 化的体素数（未知区规模；幽灵内壁排查，§4 案例 1） |
| `marching.py::marching_tetrahedra`（L185 `usable = ...`） | 含 NaN 角点的四面体比例（权重掩码是否生效，M-6） |
| `marching.py::marching_tetrahedra`（L189 `case = ...`） | 逐四面体 case ∈ [0, 15] 的分布（M-1） |
| `marching.py::marching_tetrahedra`（L202 `tt = fa[crossed] / ...`） | 交点参数 t 应全部落在 (0, 1)（M-2） |
| `marching.py::marching_tetrahedra`（L241 `uniq, first_idx, inverse = ...`） | 去重前后顶点数之比 ≈ 2（每顶点约被 2 个三角形共享）（M-4） |
| `marching.py::marching_tetrahedra`（L251 `flip = ...`） | 取向翻转的三角形数，健康 ≈ 半数（初始取向随四面体朝向随机）（M-5） |
| `metrics.py::chamfer_distance`（L90–91 `d_pq / d_qp`） | 两个单向均值应同量级；悬殊 = 一侧覆盖缺失（E-3） |
| `metrics.py::f_score`（L171–172 `precision / recall`） | 分清 P/R 各自掉了多少（§4 案例 4） |
| `demo.py::main`（L60 `volume.integrate(...)`） | 每帧 updated 返回值（T-2）；逐帧应稳定在同一量级 |
| `poisson.py::rasterize_normal_field`（L120 `wgt = np.exp(...)`） | 涂抹核权重 (K,)：窗口 7³ 内高斯值从 1 衰到 <1%（P-1） |
| `poisson.py::solve_poisson_fft`（L183 `chi_hat[nonzero] = ...`） | 频域除法一步：`lam` 在零频为 0、其余负值（P-2） |
| `poisson.py::implicit_from_points`（L259–264 `q_lo/q_hi` → `isolevel`） | 两平台分位数与样本均值水平 γ ≈ 0.5（P-3/P-4） |
| `plane_sweep.py::plane_homography`（L133 `denom = ...`） | 单应第三行：denom 在视锥内应为正、远离 0（D-1） |
| `plane_sweep.py::_window_ncc`（L186 `ncc = np.einsum(...)`） | 真实深度处的 NCC 立方体 (H-k+1, W-k+1)：盘内应接近 1（D-2） |
| `plane_sweep.py::run_plane_sweep`（L253–256 `best / ordered / confidence`） | argmin 深度与二名代价差：盘内 conf 显著大于轮廓带（D-3/D-4） |
| `splatting.py::splat_render`（L218 `mu_cam = ...`） | 相机系基元坐标：z 应全为正且 ≈ 相机距（世界→相机方向错则 z 乱，§4 案例 2 同源）（G-1） |
| `splatting.py::splat_render`（L256–263 `expo / alpha / transmittance`） | 逐 splat 的指数场、α 与透射率递推（(8.9)-(8.10) 一步全貌）（G-2/G-3） |

---

## 3. 重点观察变量表（核心章节）

约定：**形状**为断点处的 numpy 形状；**健康值**为基准场景（demo / 测试定种子）实测；
**异常信号**出现即有 bug 或配置错误。分组编号 S(cene)/T(sdf)/M(arching)/E(valuation)
/P(oisson)/D(plane-sweep depth)/G(aussian splatting)。

### S：scene（断点 `scene.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| S-1 | `render_depth` L388 | `s` | 活跃光线的当前 SDF 步长 (A,) | 首步 ≈ 相机到场景距离 ~1.0–2.5，逐步收缩到 < 1e-4 | 步长不收缩（不收敛）→ SDF 有误或光线不指向场景；步长为负 → 相机在场景内部 |
| S-2 | `render_depth` L391 | `hit` | 命中掩码 (A,) | 每步命中数从 0 升到峰值后归零 | 一步命中 100% → hit_eps 过大；永不命中 → t_max 太小 |
| S-3 | `render_depth` 返回 | `depth` | 深度图 (H, W)，单位 m | demo 口径有效像素 ~400/3136（12.5%）；中心像素 = d − r ± 2e-3 | 有效率骤降 → 位姿/内参错（如 K 主点写反）；出现负深度 → d̂_z 约定破坏 |
| S-4 | `Scene.sdf` L205 | `np.min(np.stack(...), axis=0)` | 并集 SDF (M,) | 与单基元 min 一致（`test_union_scene_min_rule` 1e-12） | 大于任一单基元值 → 并集写成了 max/加法 |
| S-5 | `make_orbit_poses`（`R_wc` 一行） | `np.linalg.det(R_wc)` | 旋转行列式（标量） | +1（right = forward × up 的右手系） | **−1 → 左手系**：深度图左右镜像、法向朝内（§4 案例 2） |
| S-6 | `render_dataset` L443 | `sigma` | 逐像素噪声标准差，单位 m | σ = 0.002·z²：z≈1.6 m 时 ≈ 5e-3 | σ 与深度无关 → 写成了常数；z=inf 处非零 → 没按 `finite` 掩码 |
| S-7 | `sample_scene_surface` L490 | `gap` | 最近/次近基元 \|sdf\| 之差 (K,)，m | > 1e-6（被保留）；基元重叠场景大量 ≈ 0 | gap 全为 0 → 基元重叠，投影歧义，采样数骤减 |

### T：tsdf（断点 `tsdf.py::TSDFVolume.integrate` 为主）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| T-1 | L146 | `s` | 离散观测 z(u) − zₓ (V,)，单位 m | 带内子集集中在 ±mu（demo mu=0.075）内；其余被 band 掩掉 | 系统性偏移 ±体素以上 → 位姿/内参错；全部为正 → 场景在相机后方 |
| T-2 | L147（或返回值） | `band` / `updated` | 带内掩码 (V,) / 本次写入体素数（int） | demo 24 帧逐帧 ~2–4e4；`int((weight>0).sum())` 融合完 = 16177 | updated = 0 → 深度全 inf 或位姿把场景投出画面；每帧写入量剧烈波动 → 深度图乱序 |
| T-3 | L157 | `w` | 单帧权重 1/z² (K,)，单位 1/m² | ≈ 0.3–0.5（z ≈ 1.4–1.6 m） | 与 z 无关 → 忘了平方；inf → z(u)=0 未过滤 |
| T-4 | L158 | `f_obs` | 截断观测 clip(s/μ) (K,)，无量纲 | [-1, 1]；近表面 |s/μ| 小 | 超出 ±1 → clip 被删；恒 ±1 → mu 过小或 s 单位错 |
| T-5 | L163–164 | `w_new` / `f_new` | (4.8) 递归后的 W 与 F (K,) | F 向 0 收敛（多视角平均）；W 单调增 | F 发散/超 ±1 → 分母写错；W 不增 → 索引 `flat` 错位（写入到错误体素） |
| T-6 | `extraction_field` L227 | `field`（NaN 掩码后） | (n, n, n)，NaN = 未观测 | demo：NaN 41407 / 262144（84%） | 全部非 NaN → weight 忘了更新；表面附近被 NaN → W 掩码条件写反 |
| T-7 | `sample_tsdf` L194 | `acc` | 三线性插值累加 (M,) | 体素中心处 = 存储值（1e-12）；棱中点 = 均值 | 中心点也对不上 → g 坐标的 0.5·voxel 偏移漏加 |

### M：marching（断点 `marching.py::marching_tetrahedra`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| M-1 | L189 | `case` | 每四面体符号配置 (C,)，0..15 | 单位球 16 case 全有出现；case 0/15 占多数 | 出现 > 15 或 < 0 → 位移位错；全 15 → 场恒正（提取了个寂寞） |
| M-2 | L202 | `tt` | 棱上插值参数 (K,)，(5.1) | 全部 ∈ (0, 1)（异号前提保证） | 超界 → `crossed` 掩码与除法用的 fa/fb 不一致；nan → fa−fb=0 同号却被放行 |
| M-3 | L210 | `edge_keys` | 全局棱键 min·N_nodes+max (K,)，int64 | 同一几何棱跨四面体键相同（去重的前提） | 顶点数 ≈ 3×三角形数（没去重）→ 键构造漏了 min/max 排序 |
| M-4 | L241 | `uniq / inverse` | 唯一键 (U,) / 逆索引 (3T,) | U ≈ V/2（demo 15076 顶点 / 29421 三角形） | 出现重复顶点（坐标相同却两个 id）→ 闭合性测试立即失败 |
| M-5 | L251 | `flip` | 取向翻转掩码 (T,) | ~50%（初始取向随机，翻转后统一朝外） | 全 False 或全 True → 梯度轴序错（z,y,x vs x,y,z，§4 案例 2 同源） |
| M-6 | L185 | `usable` | 四面体 NaN 掩码 (C,) | 单位球测试全 True；demo 中含 NaN 四面体 = 未知区相邻者 | 全 False → NaN 扩散到表面带（掩码条件写反）；无 NaN 却有幽灵面 → 忘用 extraction_field |

### E：metrics 与 demo（断点 `metrics.py` / `demo.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| E-1 | `chamfer_distance` L90–91 | `d_pq` / `d_qp` | 两个单向平均平方距离，m² | demo：0.000127 / 0.000129（同量级） | 悬殊 > 10× → 一侧覆盖缺失（acc 大 = 重建多余/偏移，comp 大 = 覆盖缺口） |
| E-2 | `f_score` L171–172 | `precision` / `recall` | 阈值内比例，无量纲 | demo：P ≈ R ≈ 0.96；单视角基线 P≈1 / R≈0.315 | P 高 R 低 → 重建"少而准"（视角不足）；P 低 R 高 → 重建"多而糙"（噪声/幽灵面） |
| E-3 | `demo.py` L75 | `cd` | 端到端 Chamfer，m² | **1.27–1.29e-4**（RMS ≈ 1.1 cm；双环境打印末位差属预期） | > 1e-3 → 幽灵内壁（§4 案例 1）；> 1e-2 → 位姿/内参错 |
| E-4 | `demo.py` L73 | `rec_points` / `gt_points` | 重建/真值表面点 (8192, 3) 各 | 两者都落在场景包围盒内；gt 的 \|sdf\| < 1e-9 | gt 的 \|sdf\| 大 → `sample_scene_surface` 的 band/seed 改动；rec 飞出包围盒 → 场错 |

### P：poisson（断点 `poisson.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| P-1 | `rasterize_normal_field` L120 | `wgt` | 涂抹核权重 (K,)（窗口 7³ 内） | 中心 1、边缘 < 1%（高斯 3σ 截断） | 全 1 → σ 单位错（体素 vs 米）；全 ≈ 0 → 点集离网格太远（pad 不足） |
| P-2 | `solve_poisson_fft` L183 | `lam` | 拉普拉斯特征值 (n, n, n) | 零频 = 0、其余 < 0（最大 \|λ\| ≈ 12/h²） | 零频非零 → fft 频率序写错；出现正值 → cos 符号错 |
| P-3 | `implicit_from_points` L259 | `q_lo / q_hi` | 解场两平台分位数（标量各） | q_hi − q_lo ≫ 0（内/外两平台分明） | 两者几乎相等 → 解退化（法向取向不一致/零场） |
| P-4 | `implicit_from_points` L264 | `isolevel` | 样本均值等值面水平 γ（标量） | ≈ 0.5（实测 0.47；容差 [0.3, 0.7]） | 远离 0.5 → 法向取向混（外向/内向混杂）或点集不在网格内 |

### D：plane sweep（断点 `plane_sweep.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| D-1 | `plane_homography` L133 | `denom` | 单应齐次分母 (H, W) | 全部同号且远离 0（视锥内 ≈ d 量级） | 变号/近零 → 深度假设为负或位姿共轭（相机重合） |
| D-2 | `_window_ncc` L186 | `ncc` | 逐像素窗口 NCC (H-k+1, W-k+1) | 真实深度假设处球盘内 → 接近 1；错误深度 → 明显偏低 | 全图处处低 → 纹理不是世界坐标函数（跨视角不同色）或位姿错 |
| D-3 | `run_plane_sweep` L253 | `best` / `depth_map` | 最优深度下标/深度 (H, W) | 球盘内与 gt 深度差 < 假设间距×2 | 系统性偏移一个假设以上 → 深度范围不含真值或 H(d) 符号错 |
| D-4 | `run_plane_sweep` L256 | `confidence` | 二名代价差 (H, W) | 盘内 ≈ 0.03–0.05；轮廓带/背景 ≈ 0（实测 0.045 vs 0.013） | 处处并列（conf ≡ 0）→ 弱纹理（常色场景）；盘内也为 0 → 成本体积全 1（NCC 判别力失效） |

### G：splatting（断点 `splatting.py::splat_render`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| G-1 | L218 | `mu_cam` | 相机系基元坐标 (M, 3) | z 全 > 0 且 ≈ 相机到场景距离（~1.25） | z 出现负值/量级错 → 世界→相机方向写反（R 与 Rᵀ 用反，§4 案例 2 同源） |
| G-2 | L256 | `expo` | 逐 splat 二次型 (v1−v0, u1−u0) | 中心 0、按椭圆向外增大；bbox 外被截断 | 全图巨大值 → Σ' 病态（忘加低通项 det≈0）；恒 0 → 逆矩阵错 |
| G-3 | L263 | `transmittance` | 透射率 (H, W) | 盘内 → 0（不透明）、盘外 = 1；随 splat 递推单调降 | 盘外也被降 → α 求值区域越界（bbox 偏移）；不减反增 → (1−α) 写成 α |

---

## 4. 常见调试场景（症状 → 断点 → 看什么 → 结论）

### 案例 1：Chamfer 突增 10 倍 + 网格顶点翻倍（幽灵内壁）

- **症状**：demo 的 Chamfer 从 1.28e-4 恶化到 ~1e-3，顶点数明显变多；可视化网格
  发现物体内部多了一层壳。
- **断点**：`tsdf.py::TSDFVolume.extraction_field`（L227）与
  `marching.py` L185（`usable`）。
- **看什么**：直接把 `volume.tsdf`（而非 `extraction_field()`）喂给提取时，
  `usable` 全 True（M-6 异常信号）；未知体素以 (F=+1, W=0) 存在，与观测带内边界
  （F≈−1）构成一个"假"符号变化面。
- **结论**：提取必须用 `extraction_field()`（W=0 → NaN，含 NaN 角点的四面体整体
  跳过）。这是 KinectFusion 实践中的权重阈值提取；`tests/test_marching.py::
  test_nan_unknown_region_skipped` 守护该行为。

### 案例 2：法向一半朝内 / 深度图左右镜像（相机基手性）

- **症状**：`test_normals_point_outward` 失败（朝外比例 ~0.75 而非 >0.95），或
  渲染深度图与直觉左右相反。
- **断点**：`scene.py::make_orbit_poses`（`R_wc = np.column_stack(...)` 一行）。
- **看什么**：`np.linalg.det(R_wc)`（S-5）。历史 bug：`right = cross(up, forward)`
  构造出**左手系**（det = −1），图像左右镜像、且梯度法向的"朝外"判定翻转
  （与 M-5 的全翻转同源）。
- **结论**：CV 相机系（x 右、y 下、z 前光轴）必须右手：`right = cross(forward, up)`、
  `down = cross(forward, right)`。修好后 `test_orbit_poses_look_at_origin` 与
  `test_normals_point_outward` 同时回绿。

### 案例 3：闭合性测试失败 / 出现重复顶点（16 case 表与去重键）

- **症状**：`test_mesh_is_closed_every_edge_shared_by_two_triangles` 报
  count ∈ {1, 3, 6...}；或网格出现顶点坐标相同的重复顶点、三角形退化。
- **断点**：`marching.py::MT_TRIANGLES` 与 L210（`edge_keys`）、L241（`np.unique`）。
- **看什么**：① `test_mt_table_matches_sign_configuration` 穷举校验每个 case 的
  三角形棱集合 == 变号棱集合——历史上 case 9/11/13 的补 case 棱集抄错过（症状是
  引用未填充的棱槽 → 键 0 → 大量顶点塌到同一点）；② `edge_keys` 是否用
  `min(na,nb)·N + max(na,nb)`（有序格点对）——跨立方体共享棱必须同键。
- **结论**：手写表先跑穷举自检再接端到端；去重键必须是整数格点对而非浮点坐标。

### 案例 4：F-score 比 demo 低一大截（评测点云密度 vs tau）

- **症状**：照 TUTORIAL §3.5 复现，只有 2000 个评测点，F@20mm 只有 ~0.67 而 demo
  是 0.96，怀疑重建坏了。
- **断点**：`metrics.py::f_score`（L171–172）与 `demo.py` L73（`rec_points`）。
- **看什么**：E-2 的 precision/recall 哪个掉了；再算采样间距——表面积 ~2.7 m²，
  2000 点 → 平均间距 ~3.7 cm > tau=20 mm，**完美重建**也会有大量点对距离超 tau。
- **结论**：tau 必须与评测密度匹配（经验：tau ≥ 2×平均采样间距）。demo 用 8192 点
  （间距 ~1.8 cm）配 tau=20 mm；2000 点时把 tau 放到 50 mm（F@50mm 实测 0.996）。
  这不是 bug，是 F-score 的密度依赖——固定 tau 报告时必须固定采样数。

### 案例 5：plane_sweep 融合点云 Chamfer 异常大（置信度过滤缺失）

- **症状**：扫描深度图喂给 `TSDFVolume` 后，重建球面 Chamfer ~0.036 m²（健康
  ~0.002），网格在球周围多出一圈碎片壳。
- **断点**：`plane_sweep.py::run_plane_sweep`（L253–256）与 `D-4` 的 `confidence`。
- **看什么**：逐像素 argmin 对**每个**像素都有输出——背景与遮挡边界的代价体积
  全是"并列的最差值 1"，argmin 挑出的深度是纯噪声；这些像素直接进 TSDF 即幽灵
  深度（背景像素的 conf ≈ 0.001，盘内 ≈ 0.03–0.05，分界清晰）。
- **结论**：融合前按置信度过滤（`depth[conf <= tau] = np.inf`，tau ≈ 0.015，
  `tsdf.integrate` 自动跳过 inf）——MVSNet 的 P(d) 过滤与 COLMAP 几何一致性
  做的是同一件事。`tests/test_plane_sweep.py::test_fused_point_cloud_beats_single_view`
  守护该行为（0.0019 vs 0.018 m²）。

---

## 5. 测试怎么写

新测试放在 `tests/test_<模块名>.py`，遵循本套件既有惯例：

1. **命名 `test_<行为>`**——名字说清"验证什么行为"，因为单点过滤按子串匹配
   （§1.1），名字是过滤键。范例：`test_normals_point_outward`、
   `test_zero_crossing_near_radius`。
2. **deterministic seed**——一切随机性走 `np.random.default_rng(seed)`，使打印
   数字与断言阈值可精确复现；禁止 `np.random.seed`/全局随机态。
3. **断言阈值旁注明理由**——照套件惯例（"容差 ±1 体素"、"1e-10 交叉验证"等
   注释）写清期望值的量级来源，让后来者能区分"阈值松了"与"实现错了"。
4. **文件尾部带单点过滤主入口**——照抄任一现有测试文件的 `__main__` 块：

   ```python
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
   ```

   （文件头部还需 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
   使从任意 cwd 直跑可用，见 §1.4。）

5. **运行时长预算 < 15 s / 文件**——当前最慢的是 `tests/test_plane_sweep.py`
   （~9.8 s，含 6 参考视角的端到端融合）；大网格/大扫描通过降分辨率、减深度假设数
   而非删断言来控制时长。
   指标断言从 `metrics.py` 取实现（[METRICS.md](METRICS.md)），朴素实现只用于
   交叉验证，勿在断言里重写公式。
6. **写完自查**：`python tests/test_<新文件>.py` 全绿 + 该模块断点处的观察量
   （§3 变量表）落在健康值内 + 若新增了被监视变量，同步更新本文件的变量表
   （CONSTRAINTS §4.5 质量要求）。
