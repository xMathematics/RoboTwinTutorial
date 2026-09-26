# NeRF 测评指标教程（METRICS.md）

> **适用对象**：零基础读者——不假设你有图像处理 / 深度学习背景，术语首现均给中文通俗解释。
> **统一测评模块**：[`nerf/metrics.py`](nerf/metrics.py)（纯函数、无状态，被
> [`run_nerf.py`](run_nerf.py) 的 test 模式调用，也可独立对两张图打分）。
> **本项目指标基线**：PSNR / SSIM（LPIPS 列为延伸，未实现，见文末）。
> **论文对照**：NeRF 论文（Mildenhall et al., ECCV 2020）实验节以 PSNR / SSIM / LPIPS
> 三指标报告结果，详见教程 [06_数据集与实验结果.md](../tutorials/nerf/06_数据集与实验结果.md)。

---

## 0. 指标速查表

| 指标 | 作用（一句话） | 函数 | 单位 |
|------|---------------|------|------|
| **PSNR** 峰值信噪比 | 像素级重建保真度：渲染图与真值逐像素差多大 | `nerf/metrics.py::psnr` | dB（分贝，越大越好） |
| **SSIM** 结构相似性 | 局部亮度 / 对比度 / 结构与真值像不像（更接近人眼） | `nerf/metrics.py::ssim` | 无量纲（越大越好） |

两个指标都在 test 模式下逐张测试图计算，均值写入 `logs/<实验名>/metrics.json`。

---

## 1. PSNR（峰值信噪比，Peak Signal-to-Noise Ratio）

### 1.1 作用

- **衡量什么**：像素级重建保真度。把渲染图和真值图叠在一起逐像素比较颜色差
  多大——差异越小，PSNR 越高。NeRF 的训练目标（论文 Eq.7，均方误差 MSE）
  正是"逐像素差最小化"，PSNR 就是 MSE 换算成对数刻度（dB）后的读数，两者
  一一对应：MSE 每缩小 10 倍，PSNR 恰好升高 10 dB。
- **为什么需要它**：训练时看的是训练集损失，评估时要在**没见过的测试视角**上
  给出一个客观、可复现、可跨论文比较的数字——PSNR 承担这个角色。
- **对应论文/教程的主张**：NeRF 论文 Tab.1 / Tab.2（实验节 Sec.6）的主指标；
  三大数据集上的 PSNR 对比表见教程
  [06_数据集与实验结果.md §6.2](../tutorials/nerf/06_数据集与实验结果.md)
  （如 Realistic Synthetic 360° 上 NeRF 31.01 dB，比第二名 NV 高 4.96 dB）。

### 1.2 如何计算

先算**均方误差**（MSE，Mean Squared Error：所有像素、所有颜色通道上
"预测值 − 真值"的平方的平均），再取对数：

$$
\mathrm{MSE} = \frac{1}{HWC}\sum_{i,j,c}\big(\hat{I}_{ijc} - I_{ijc}\big)^2,\qquad
\mathrm{PSNR} = 10\cdot\log_{10}\!\Big(\frac{\mathrm{MAX}^2}{\mathrm{MSE}}\Big). \tag{M.1}
$$

- 本项目图像已归一化到 $[0,1]$，故像素上限 $\mathrm{MAX}=1$，(M.1) 退化为
  $\mathrm{PSNR}=10\cdot\log_{10}(1/\mathrm{MSE})$。
- **代码位置**：`nerf/metrics.py::psnr`（两行实现：`mse = np.mean((img1 - img2) ** 2)` →
  `10.0 * np.log10(1.0 / (mse + 1e-10))`）。分母的 `1e-10` 是数值稳定项：两图
  完全相同（MSE = 0）时防止除零得到 inf，副作用是完美重建被截断为
  $10\cdot\log_{10}(10^{10}) = 100$ dB 的**实现上限**。
- **出处**：(M.1) 是信号处理中峰值信噪比的标准定义；以 PSNR 评价新视角合成
  是 NeRF 论文实验节（及此前工作）通行的协议。
- **互引**：各数据集的 PSNR 参考值与"读表方法论"见
  [06_数据集与实验结果.md](../tutorials/nerf/06_数据集与实验结果.md) §6.2 / §6.5；
  训练侧 MSE 损失的推导见教程 [05_层次采样与训练细节.md](../tutorials/nerf/05_层次采样与训练细节.md)。

### 1.3 健康值范围

| 场景 | PSNR 实测 / 参考 | 说明 |
|------|-----------------|------|
| **demo_scene 冒烟（本项目实测）** | **16.23 dB**（默认 testskip=8，评 1 张）／ **19.23 dB**（`--testskip 1`，评全部 8 张） | 80 步 tiny 训练（`--tiny --steps 80`，约 4 秒），仅证明管线通畅、损失在降（训练日志 loss 0.2024 → 0.0247，MSE≈0.01~0.02 ⇒ PSNR 17~20 dB，量级自洽） |
| **lego 等真实数据集（未实测，如实说明）** | 论文全量训练约 30 dB 量级（Realistic Synthetic 360° 八场景平均 31.01 dB / SSIM 0.947，见教程 06 章 §6.2） | 本仓库不附带 lego 数据，未做实测；tiny 档几千步的冒烟训练在真实数据上通常只有 20 dB 上下，**不要拿冒烟值对标论文值**——两者训练步数差 3 个数量级 |
| 完美重建（实现上限） | 恰为 100.0 dB | `1e-10` 稳定项造成，见 §1.2 |

**异常信号**（真实数据、正常配置训练后）：

| 现象 | 可能原因 |
|------|---------|
| PSNR < 15 dB | 训练远未收敛 / 步数太少 / 学习率不当 |
| PSNR 恰为 100 dB（SSIM 恰为 1.0） | 误把真值图本身当成了预测图去评估（数据加载或路径错误） |
| PSNR 与训练损失严重不匹配（损失很低、PSNR 也低） | 测试集与训练集视角分布差异过大，或位姿/内参解读错误 |

### 1.4 如何运行

**① 单元测试**（在 `projects/nerf/` 下，conda 环境 `llm_env`；已实跑验证）：

```bash
conda activate llm_env
python tests/test_metrics.py            # 全部 7 项，约 2 秒
python tests/test_metrics.py ssim       # 单点：只跑名字含 ssim 的 3 项
```

全部通过时输出：

```
PASS  test_equivalence_with_original_inline_implementation
PASS  test_psnr_identical_images_caps_at_100db
PASS  test_psnr_matches_naive_mse_formula
PASS  test_psnr_monotonic_decreases_with_noise
PASS  test_ssim_identical_images_is_one
PASS  test_ssim_matches_naive_window_implementation
PASS  test_ssim_structure_vs_luminance_perturbation

7/7 tests passed
```

**② 端到端评测**（生成演示数据 → tiny 训练 80 步 → test 模式打分；已实跑验证）：

```bash
python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32
python run_nerf.py --config data/demo_scene --mode train --exp demo_smoke --tiny --steps 80
# 默认 testskip=8：8 张测试图每 8 张取 1 张 → 评 1 张，秒级
python run_nerf.py --config data/demo_scene --mode test --exp demo_smoke --ckpt logs/demo_smoke/latest.pt
# 加 --testskip 1：评全部 8 张测试图
python run_nerf.py --config data/demo_scene --mode test --exp demo_smoke --ckpt logs/demo_smoke/latest.pt --testskip 1
```

**输出解读**：每张测试图的渲染 PNG 存到 `logs/demo_smoke/test/pred_XXX.png`，
最后一行汇总打印（本次实测值）：

```
[test] 8 test images @ 16x16
[test] PSNR 19.23 | SSIM 0.8346
```

同时写入 `logs/demo_smoke/metrics.json`，字段：`psnr_mean` / `ssim_mean`
（均值）与 `psnr_list` / `ssim_list`（逐张列表，用于画逐图分布、找最差视角）。

**③ 独立打分**（统一测评模块可不经过训练管线直接用；已实跑验证）：

```bash
python -c "
import imageio.v2 as imageio
import numpy as np
from nerf.metrics import psnr, ssim
pred = imageio.imread('logs/demo_smoke/test/pred_000.png').astype(np.float32) / 255.0
gt = imageio.imread('data/demo_scene/test/r_000.png').astype(np.float32) / 255.0
gt = gt[::2, ::2]   # half_res 评测约定：真值同样隔点降采样到 16x16
print('PSNR %.2f dB | SSIM %.4f' % (psnr(pred, gt), ssim(pred, gt)))
"
# 实测输出：PSNR 16.23 dB | SSIM 0.7678
```

> 说明：读 PNG 打分与 test 模式的内存内打分有 ≤0.001 量级的小差异（SSIM
> 0.7678 vs 0.7682），因为 PNG 保存时做了 uint8 量化；PSNR 在两位小数内一致。
> 官方 nerf_synthetic 图像带 alpha 通道，需按白底合成（用
> `nerf/data_utils.py::_read_image`），而 demo_scene 的 PNG 是纯 RGB 可直接读。

---

## 2. SSIM（结构相似性，Structural Similarity）

### 2.1 作用

- **衡量什么**：**结构相似性**——不看逐像素的绝对误差，而是对图像的每个局部
  窗口比较三件事：**亮度**（窗口内平均亮度像不像）、**对比度**（亮暗起伏的
  幅度像不像）、**结构**（亮暗的空间分布 pattern 像不像），合成一个 0~1 的
  相似度。名字的由来正在于此：它显式地把"结构"从亮度、对比度里拆出来单独比。
- **为什么需要它**：PSNR 对所有像素一视同仁，人眼却不是——整体偏亮一点几乎
  察觉不到，但纹理错位、边缘模糊一眼就能看出。SSIM 与人眼判断的相关性显著
  高于 PSNR（这是 Wang et al. 2004 论文用大规模主观实验验证过的主张），所以
  PSNR 之外还要报 SSIM，两个互补：一个客观便宜、一个贴近感知。
- **为什么 PSNR 和 SSIM 都要报**：只看 PSNR 可能漏掉"数值误差小但结构错位"
  的坏情况；只看 SSIM 又丢失了对绝对颜色的约束（渲染图整体色偏时 SSIM 可能
  仍很高）。NeRF 论文 Tab.1 / Tab.2 两者同时报告，本项目与论文保持一致。
- **对应论文/教程的主张**：NeRF 论文在三大数据集上均报告 SSIM（如 Realistic
  Synthetic 360° 上 0.947），对比表见
  [06_数据集与实验结果.md §6.2](../tutorials/nerf/06_数据集与实验结果.md)。

### 2.2 如何计算

对图中每个像素取一个高斯加权窗口（本实现为标准差 1.5、截断为 13×13 的高斯
核，即"离窗口中心越近的像素权重越大"），在窗口内估计两图的均值
$\mu_x,\mu_y$、方差 $\sigma_x^2,\sigma_y^2$、协方差 $\sigma_{xy}$，再按：

$$
\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)\,(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)\,(\sigma_x^2 + \sigma_y^2 + C_2)},\qquad
C_1=(k_1 L)^2,\ C_2=(k_2 L)^2. \tag{M.2}
$$

- 三个因子对应：$\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}$（亮度）、
  $\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}$（对比度，在 (M.2)
  中与结构项合并出现）、$\frac{\sigma_{xy}}{\sigma_x\sigma_y}$（结构，分子中的
  $2\sigma_{xy}$ 即其核心）。$C_1, C_2$ 是分母稳定常数，防止局部方差为 0 时
  除零；$L$ 为像素动态范围（数据已归一化到 $[0,1]$，故 $L=1$；
  $k_1=0.01,\ k_2=0.03$）。
- **代码位置**：`nerf/metrics.py::ssim`——逐窗口统计用
  `scipy.ndimage.gaussian_filter` 的全图可分离卷积实现（数学上与 Wang 原文的
  逐窗计算等价，等价性由 `tests/test_metrics.py` 的朴素双循环实现交叉验证，
  实测差异 ~1e-16），最后对整图 SSIM 图取均值得到单值。
- **出处**：Wang, Bovik, Sheikh, Simoncelli, *"Image quality assessment: from
  error visibility to structural similarity"*, IEEE Trans. Image Process. 13(4),
  2004——三因子定义为其 Eq. 6，高斯加权统计量为其 Eq. 7–9，(M.2) 为指数取 1
  的简化式（其 Eq. 13）；常数 $\sigma=1.5, k_1=0.01, k_2=0.03$ 为该文推荐设置。
- **互引**：SSIM 在 NeRF 论文结果表中的位置与"LLFF 在 LPIPS 上反超"的读表
  细节见 [06_数据集与实验结果.md](../tutorials/nerf/06_数据集与实验结果.md)
  §6.2；SSIM 与 PSNR 的互补关系见本文件 §2.1。

### 2.3 健康值范围

| 场景 | SSIM 实测 / 参考 | 说明 |
|------|-----------------|------|
| **demo_scene 冒烟（本项目实测）** | **0.7682**（默认 testskip=8，评 1 张）／ **0.8346**（`--testskip 1`，评全部 8 张） | 与 PSNR 同一次 80 步 tiny 训练；结构尚未学精细（16×16 低分辨率 + 极少步数），0.77~0.83 属正常冒烟水平 |
| **lego 等真实数据集（未实测，如实说明）** | 论文全量训练 0.9~0.95 量级（Realistic Synthetic 360° 平均 0.947） | 本仓库未附带 lego 数据；冒烟档在真实数据上通常 0.5~0.8，与论文值不可比（步数差 3 个数量级） |
| 完全相同的两张图 | 恰为 1.0 | 实现上限（理论范围 $[-1,1]$ 的右端点） |

**异常信号**：

| 现象 | 可能原因 |
|------|---------|
| SSIM ≈ 0 或为负 | 渲染图与真值在局部上不相关——位姿/内参错误、场景对不上，比低 PSNR 更能暴露"根本没学到这个场景" |
| PSNR 尚可（>20 dB）但 SSIM 明显偏低 | 渲染结果整体发糊/伪影多：像素均值对了但结构错了，正是 SSIM 想抓的情况 |
| SSIM 恰为 1.0 且 PSNR = 100 dB | 误把真值当预测评估 |

**结构 vs 亮度的区分度**（`tests/test_metrics.py::test_ssim_structure_vs_luminance_perturbation`
实测）：把图像整体加亮 0.05，SSIM 只从 1.0 跌到 0.98+（亮度项被 $C_1$ 兜底，
几乎无损）；而把图像水平翻转（每个像素的值都没变，只挪了位置），SSIM 直接
跌破 0.5——这就是"结构相似性"区别于逐像素指标的核心性质。

### 2.4 如何运行

- 评测命令与 §1.4 完全相同（test 模式同时输出两个指标，无需另跑）；
  SSIM 的单元测试单点运行：

```bash
python tests/test_metrics.py ssim     # 实测：3/3 tests passed
```

- 输出解读：`[test] PSNR 19.23 | SSIM 0.8346` 中 SSIM 为全部测试图的均值
  （无量纲，保留 4 位小数）；逐张值在 `metrics.json` 的 `ssim_list`。

---

## 3. PSNR / SSIM / LPIPS 的关系

一句话：**PSNR 只看逐像素误差（客观、便宜、对结构失真不敏感）→ SSIM 在局部
窗口里比较亮度/对比度/结构（更接近人眼）→ LPIPS 用预训练深度网络的特征距离
衡量感知差异（最接近人眼判断，但需要额外网络）**，三者沿着"像素 → 结构 →
语义感知"逐级逼近人类视觉判断，因此 NeRF 论文 Tab.1/Tab.2 同时报告三者。
其中 **LPIPS（Learned Perceptual Image Patch Similarity，Zhang et al., CVPR
2018）在本项目中被列为延伸、未实现**——需要引入预训练网络权重；若需复现
论文完整指标可参考其[官方实现](https://github.com/richzhang/PerceptualSimilarity)。

---

## 附：文件与命令速查

| 想做什么 | 去哪里 |
|---------|--------|
| 看指标实现 | [`nerf/metrics.py`](nerf/metrics.py)（中文 docstring 含公式出处） |
| 跑指标测试 | `python tests/test_metrics.py`（全部）/ `python tests/test_metrics.py ssim`（单点） |
| 端到端评测 | `python run_nerf.py --config <场景> --mode test --exp <实验名> --ckpt <ckpt路径>` |
| 看结果 | `logs/<实验名>/metrics.json` + `[test]` 汇总行 |
| 论文结果对照 | [06_数据集与实验结果.md](../tutorials/nerf/06_数据集与实验结果.md) |
