"""统一测评模块：新视角合成质量的图像级指标（PSNR / SSIM）。

函数流水线：本模块是项目的**统一测评层**，处于管线最末端的指标侧——
    run_nerf.py（test 模式）→ 逐张 render_image 得到渲染图 pred（[H, W, 3]）
        → metrics.psnr(pred, gt)   逐图峰值信噪比（标量，dB）
        → metrics.ssim(pred, gt)   逐图结构相似性（标量，无量纲）
    → 汇总为均值写入 logs/<exp>/metrics.json 并打印（PSNR xx.xx | SSIM x.xxxx）。
本模块为**纯函数、无状态、无 I/O**：不依赖 torch 与项目内其他模块，只吃 numpy
数组，因此未来任何评估脚本（如对两张 PNG 离线打分、批量评测 checkpoint）都
可直接复用，而不必经由训练管线。

公式出处：
    PSNR —— 信号处理中峰值信噪比的标准定义（MSE 换算到 dB）；NeRF 论文
            （Mildenhall et al., ECCV 2020）实验节（Sec. 6）以 PSNR/SSIM/LPIPS
            作为新视角合成的三大定量指标（结果见教程 06_数据集与实验结果.md）。
    SSIM —— Wang, Bovik, Sheikh, Simoncelli, "Image quality assessment: from
            error visibility to structural similarity", IEEE Trans. Image
            Process. 13(4), 2004：局部亮度/对比度/结构三因子（Eq. 6），
            高斯加权统计量（Eq. 7-9），指数为 1 的简化乘积式（Eq. 13）。
"""
import numpy as np


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """峰值信噪比 PSNR（Peak Signal-to-Noise Ratio），衡量像素级重建保真度。

    数学含义：PSNR = 10·log10(MAX²/MSE)，其中 MAX 为像素动态范围上限。本实现
    约定图像已归一化到 [0, 1]，故 MAX = 1，公式退化为 10·log10(1/MSE)。它是
    NeRF 论文实验节（Tab. 1 / Tab. 2）的像素级主指标。分母加 1e-10 是数值
    稳定项：两图完全相同（MSE = 0）时 10·log10(1/0) 发散为 +inf，加地板项后
    截断为 10·log10(1e10) = 100 dB 的实现上限。

    Args:
        img1: 预测图像，形状 [H, W, 3]，取值 ∈ [0, 1]（无量纲，已按 255 归一化）。
        img2: 参考真值图像，形状 [H, W, 3]（与 img1 同尺寸），取值 ∈ [0, 1]。

    Returns:
        PSNR 标量，单位 dB（分贝）。取值范围：本实现下 ∈ [-∞, 100]，实际
        MSE ≤ 1 时 ≥ 0；越大越好。经验参考：> 20 dB 可辨轮廓，> 30 dB 质量
        较好（NeRF 论文 Realistic Synthetic 360° 上为 31.01 dB）。
    """
    # 依据：PSNR 定义 10·log10(MAX²/MSE)，数据范围 [0,1] ⇒ MAX²=1
    mse = np.mean((img1 - img2) ** 2)
    # 依据：数值稳定项 1e-10——MSE=0（完美重建）时防止除零得到 inf，截断为 100 dB
    return float(10.0 * np.log10(1.0 / (mse + 1e-10)))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """结构相似性 SSIM（高斯窗口的简化实现，依据 Wang et al. 2004 的定义）。

    数学含义：SSIM 把图像质量建模为**亮度、对比度、结构**三个局部因子的乘积
    （Wang et al. 2004 Eq. 6）；对每个像素取局部窗口，用高斯核加权估计两图的
    均值 μ（亮度）、方差 σ²（对比度）与协方差 σ12（结构）（Eq. 7-9），再按
    指数为 1 的简化乘积式合成逐像素 SSIM 图（Eq. 13，本实现的最后一行均值前
    的分式）::

        SSIM(x, y) = (2·μx·μy + C1)(2·σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))

    其中 C1 = (k1·L)²、C2 = (k2·L)² 为分母稳定常数（防止局部方差为 0 时除零），
    L 为像素动态范围（数据已归一化到 [0, 1]，故 L = 1）；局部统计量用标准差
    1.5 的高斯核加权（Wang et al. 2004 推荐 σ=1.5、k1=0.01、k2=0.03），窗口由
    scipy 默认 truncate=4.0 截断为 13×13。与 Wang 原文逐窗计算不同，这里用
    scipy.ndimage.gaussian_filter 对整图做可分离卷积（数学上等价，数值见
    tests/test_metrics.py 的朴素实现交叉验证）。最后对整图 SSIM 图取均值得到
    单值指标。

    Args:
        img1: 预测图像，形状 [H, W, 3]，取值 ∈ [0, 1]（无量纲，已按 255 归一化）。
        img2: 参考真值图像，形状 [H, W, 3]（与 img1 同尺寸），取值 ∈ [0, 1]。

    Returns:
        SSIM 标量，无量纲。理论取值范围 [-1, 1]：1 = 完全相同（本实现相同输入
        精确返回 1.0），负值仅在局部反相关时出现；实际场景中 0~1，越大越相似。
        NeRF 论文 Realistic Synthetic 360° 上为 0.947。
    """
    from scipy.ndimage import gaussian_filter

    # SSIM 常数：k1, k2 为经验稳定系数，L 为像素动态范围（此处已归一化到 1）
    # 依据：Wang et al. 2004 Eq.(13)，C1=(k1·L)²、C2=(k2·L)²
    k1, k2, L = 0.01, 0.03, 1.0
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    # 局部均值 μ（亮度统计量；依据：Eq.(7) 高斯加权平均，σ=1.5）
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    # 局部方差 σ² 与协方差 σ12（对比度/结构统计量；依据：Eq.(8)(9)，σ²=E[x²]−μ²）
    s1_sq = gaussian_filter(img1 * img1, 1.5) - mu1_sq
    s2_sq = gaussian_filter(img2 * img2, 1.5) - mu2_sq
    s12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2
    # 逐像素 SSIM 图（依据：Eq.(13)，取幂次 α=β=γ=1 的简化乘积式）
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    )
    return float(np.mean(ssim_map))
