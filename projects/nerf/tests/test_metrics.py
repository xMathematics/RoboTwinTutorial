"""统一测评模块（nerf/metrics.py）的单元测试。

可用 pytest 运行，也可直接执行。**deterministic**：全部随机数据来自固定种子
的 np.random.default_rng，重跑结果逐位一致。主入口支持子串过滤（单点测试）：
    python tests/test_metrics.py            # 全部测试
    python tests/test_metrics.py ssim       # 单点：只跑名字含 ssim 的测试
    pytest tests/test_metrics.py -v         # 等价的 pytest 跑法

覆盖范围：
    1. PSNR——与朴素 MSE 公式交叉验证、已知 MSE → 精确 dB 值、相同图像的
       上限处理、加噪图像 PSNR 单调下降；
    2. SSIM——相同图像 = 1、与朴素逐窗/双循环实现交叉验证（Wang et al. 2004
       Eq.7-13 的直接展开）、结构扰动比亮度扰动降得更多；
    3. 迁移等价性——与抽取前 run_nerf.py 的内联实现逐位一致（< 1e-12）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nerf.metrics import psnr, ssim  # noqa: E402


# ---------------------------------------------------------------------------
# 朴素参考实现（测试专用）：不依赖 nerf.metrics，直接从定义展开，用于交叉验证
# ---------------------------------------------------------------------------

def _naive_gaussian_filter(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """朴素高斯滤波：显式构造一维高斯核 + 逐像素双循环累加 + reflect 边界。

    与 scipy.ndimage.gaussian_filter(img, sigma)（默认 truncate=4.0、
    mode='reflect'）算法等价：
    - 半窗宽 lw = int(truncate·sigma + 0.5) = int(4·1.5 + 0.5) = 6（核长 13）；
    - 一维核 phi(x) = exp(-x²/(2σ²))，归一化（依据：核积分须为 1，常数场滤波后不变）；
    - 沿输入的每一个轴（含 [H, W, 3] 的通道轴）依次做一维相关滤波（可分离）；
    - 边界用对称反射：索引 i 按周期 2n 的三角波映射回 [0, n)。
    """
    truncate = 4.0                                    # scipy 默认截断：核覆盖 σ±4σ
    lw = int(truncate * sigma + 0.5)                  # 半窗宽 = 6
    x = np.arange(-lw, lw + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * x * x / (sigma * sigma))   # 依据：Wang et al. 2004 Eq.(7) 的高斯权重
    kernel = kernel / kernel.sum()                    # 归一化，保证常数场不变
    work = img.astype(np.float64)
    for axis in range(work.ndim):                     # 可分离滤波：逐轴做一维相关
        n = work.shape[axis]
        tmp = np.empty_like(work)
        for pos in np.ndindex(work.shape):            # 双循环：逐输出元素 × 逐抽头累加
            acc = 0.0
            for k in range(kernel.size):
                i = pos[axis] + (k - lw)
                m = i % (2 * n)                       # reflect 边界：-1→0、n→n-1，周期 2n
                i = m if m < n else 2 * n - 1 - m
                idx = list(pos)
                idx[axis] = i
                acc += kernel[k] * work[tuple(idx)]
            tmp[pos] = acc
        work = tmp
    return work


def _naive_ssim_separable(img1: np.ndarray, img2: np.ndarray) -> float:
    """朴素 SSIM（可分离双循环版）：统计量逐式对应 metrics.ssim，
    仅把 scipy 向量化高斯滤波替换为 _naive_gaussian_filter。"""
    k1, k2, L = 0.01, 0.03, 1.0                       # 常数与实现逐字一致（Eq.13：C=(k·L)²）
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu1 = _naive_gaussian_filter(img1, 1.5)
    mu2 = _naive_gaussian_filter(img2, 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = _naive_gaussian_filter(img1 * img1, 1.5) - mu1_sq   # 依据：Eq.(8) σ²=E[x²]−μ²
    s2_sq = _naive_gaussian_filter(img2 * img2, 1.5) - mu2_sq
    s12 = _naive_gaussian_filter(img1 * img2, 1.5) - mu1_mu2    # 依据：Eq.(9) 协方差
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    )                                                  # 依据：Eq.(13)
    return float(np.mean(ssim_map))


def _naive_ssim_per_window(img1: np.ndarray, img2: np.ndarray, sigma: float = 1.5) -> float:
    """朴素 SSIM（真·逐窗版）：对每个像素显式取出以它为中心的 13×13 窗口，
    用二维高斯权重（一维核的外积；依据：高斯核可分离）加权累计 μ、σ²、σ12，
    再按 Wang et al. 2004 Eq.(13) 合成该窗口的 SSIM。
    输入须为单通道 [H, W, 1]：此时通道轴长度 1，高斯滤波为恒等，
    逐窗定义与 Wang 原文完全一致。"""
    truncate = 4.0
    lw = int(truncate * sigma + 0.5)
    x = np.arange(-lw, lw + 1, dtype=np.float64)
    k1d = np.exp(-0.5 * x * x / (sigma * sigma))
    k1d = k1d / k1d.sum()
    w2d = np.outer(k1d, k1d)                          # 二维高斯权重 = 一维核外积
    a = img1.astype(np.float64)[..., 0]
    b = img2.astype(np.float64)[..., 0]
    H, W = a.shape
    k1, k2, L = 0.01, 0.03, 1.0
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2

    def reflect(i: int, n: int) -> int:
        # 对称反射边界（与 _naive_gaussian_filter 同一映射）
        m = i % (2 * n)
        return m if m < n else 2 * n - 1 - m

    ssim_map = np.zeros((H, W))
    for i in range(H):                                # 双循环：逐窗口计算
        for j in range(W):
            rows = [reflect(i + o, H) for o in range(-lw, lw + 1)]
            cols = [reflect(j + o, W) for o in range(-lw, lw + 1)]
            wa = a[np.ix_(rows, cols)]                # 以 (i, j) 为中心的 13×13 窗口
            wb = b[np.ix_(rows, cols)]
            mu_a = float(np.sum(w2d * wa))            # 依据：Eq.(7) 高斯加权均值
            mu_b = float(np.sum(w2d * wb))
            var_a = float(np.sum(w2d * wa * wa)) - mu_a * mu_a   # 依据：Eq.(8)
            var_b = float(np.sum(w2d * wb * wb)) - mu_b * mu_b
            cov = float(np.sum(w2d * wa * wb)) - mu_a * mu_b     # 依据：Eq.(9)
            ssim_map[i, j] = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
                (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
            )                                          # 依据：Eq.(13)
    return float(np.mean(ssim_map))


def _reference_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """原 run_nerf.py 内联 psnr（2026-09 抽取到 nerf/metrics.py 之前的原文），
    公式与常数逐字复制，作为迁移等价性的基准。"""
    mse = np.mean((img1 - img2) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + 1e-10)))


def _reference_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """原 run_nerf.py 内联 ssim（抽取前原文），逐字复制作为迁移基准。"""
    from scipy.ndimage import gaussian_filter

    k1, k2, L = 0.01, 0.03, 1.0
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = gaussian_filter(img1 * img1, 1.5) - mu1_sq
    s2_sq = gaussian_filter(img2 * img2, 1.5) - mu2_sq
    s12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    )
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def test_psnr_matches_naive_mse_formula():
    """PSNR 与朴素 MSE 公式交叉验证 + 已知 MSE → 精确 dB 值。

    依据：PSNR = 10·log10(MAX²/MSE)，数据范围 [0,1] ⇒ MAX=1。
    注意实现含 1e-10 数值稳定项：与"含稳定项的精确公式"比较用 1e-9 容差；
    与教科书理想值比较时稳定项引入 ~1e-10/MSE 量级的固有偏差
    （MSE=0.01 时约 4.34e-8 dB，故理想值断言放宽到 1e-6）。
    """
    # (1) 已知值：MSE ≈ 0.01（均匀差 0.1）→ 理想 20 dB
    img_a = np.zeros((8, 8, 3))
    img_b = np.full((8, 8, 3), 0.1)
    mse = float(np.mean((img_a - img_b) ** 2))         # 朴素 MSE：独立于实现复算
    expected = 10.0 * np.log10(1.0 / (mse + 1e-10))    # 精确 dB 值（含实现的 1e-10 稳定项）
    assert abs(psnr(img_a, img_b) - expected) < 1e-9
    assert abs(psnr(img_a, img_b) - 20.0) < 1e-6       # 理想公式值（稳定项偏差 ~4.3e-8 dB）

    # (2) 已知值：MSE = 1（全 1 vs 全 0）→ 精确 0 dB（稳定项偏差仅 ~4.3e-10 dB）
    ones, zeros = np.ones((4, 4, 3)), np.zeros((4, 4, 3))
    assert abs(psnr(ones, zeros) - 0.0) < 1e-9

    # (3) 随机图像对：与朴素公式 10·log10(1/(MSE+1e-10)) 逐对一致
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.random((12, 12, 3))
        b = np.clip(a + rng.normal(0.0, 0.1, a.shape), 0.0, 1.0)
        mse = float(np.mean((a - b) ** 2))
        assert abs(psnr(a, b) - 10.0 * np.log10(1.0 / (mse + 1e-10))) < 1e-9


def test_psnr_identical_images_caps_at_100db():
    """相同图像 → 实现上限 100 dB（完美重建的 PSNR 上限处理）。

    数学上 MSE=0 ⇒ PSNR=+inf；实现用 1e-10 稳定项防除零，副作用是
    完美重建被截断为 10·log10(1/1e-10) = 100 dB。docstring 说明：
    这是**实现上限**而非理论值，读指标时 100 dB 即"逐位完美重建"。
    """
    rng = np.random.default_rng(2)
    img = rng.random((10, 10, 3))
    assert abs(psnr(img, img) - 100.0) < 1e-9


def test_psnr_monotonic_decreases_with_noise():
    """加噪图像的 PSNR 随噪声幅度单调下降（保真度指标的必备性质）。"""
    rng = np.random.default_rng(7)
    base = np.linspace(0.2, 0.8, 16 * 16 * 3).reshape(16, 16, 3)  # 平滑渐变参考图
    values = []
    for sigma in [0.02, 0.05, 0.1, 0.2]:
        noisy = np.clip(base + rng.normal(0.0, sigma, base.shape), 0.0, 1.0)
        values.append(psnr(noisy, base))
    # 无噪声（完美重建）= 上限 100 dB，应高于一切加噪情形
    assert values[0] < psnr(base, base)
    # 噪声越大 PSNR 严格越小
    assert all(a > b for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def test_ssim_identical_images_is_one():
    """相同图像 → SSIM = 1（实现上限处理）。

    相同输入下分子分母逐项相同（2·μ1·μ2 = μ1²+μ2² 等），逐像素商精确为 1，
    均值仍为 1——即"结构完全相似"的实现上限（理论范围 [-1, 1] 的右端点）。
    """
    rng = np.random.default_rng(3)
    img = rng.random((12, 12, 3))
    assert abs(ssim(img, img) - 1.0) < 1e-12


def test_ssim_matches_naive_window_implementation():
    """SSIM 与朴素逐窗/双循环实现交叉验证（16×16 小图，< 1e-9）。

    两个方向的朴素实现（详见各 helper docstring）：
    (a) 可分离双循环版（[16,16,3]，与 run_nerf.py 生产调用形状一致）：
        双循环显式高斯滤波 vs 实现的 scipy 向量化；
    (b) 真·逐窗版（[16,16,1]，Wang 原文的窗口定义）：逐像素取 13×13 窗口
        二维高斯加权统计 vs 实现的全图可分离卷积。
    实测差异 ~1e-16（浮点结合序不同），远小于 1e-9 容差。
    """
    rng = np.random.default_rng(1)
    # (a) 三通道：生产路径形状
    a3 = rng.random((16, 16, 3)) * 0.6 + 0.2
    b3 = np.clip(a3 + rng.normal(0.0, 0.05, a3.shape), 0.0, 1.0)
    assert abs(ssim(a3, b3) - _naive_ssim_separable(a3, b3)) < 1e-9
    # (b) 单通道：真·逐窗定义
    a1 = rng.random((16, 16, 1)) * 0.6 + 0.2
    b1 = np.clip(a1 + rng.normal(0.0, 0.05, a1.shape), 0.0, 1.0)
    assert abs(ssim(a1, b1) - _naive_ssim_per_window(a1, b1)) < 1e-9


def test_ssim_structure_vs_luminance_perturbation():
    """结构扰动比亮度扰动使 SSIM 降得更多——"结构相似性"名字的由来。

    Wang et al. 2004 的核心主张：人眼对**结构**失真远比对亮度/对比度失真
    敏感。SSIM 三因子中，亮度项 (2μ1μ2+C1)/(μ1²+μ2²+C1) 对整体加亮很宽容
    （分子分母同步变大，仅略小于 1；方差与协方差在常数平移下不变，故
    对比度/结构项仍为 1）；而结构项依赖局部协方差 σ12——把图像水平翻转后
    逐像素值不变但局部结构完全失配，σ12≈0，SSIM 大幅跌落。
    """
    rng = np.random.default_rng(0)
    base = rng.random((16, 16, 3)) * 0.6 + 0.2         # [0.2, 0.8]，加亮 0.05 后仍在 [0,1]
    lum = base + 0.05                                   # 亮度扰动：整体加亮
    struct = base[:, ::-1, :].copy()                    # 结构扰动：水平翻转（值不变、位置打乱）
    s_lum = ssim(base, lum)
    s_struct = ssim(base, struct)
    assert s_lum < 1.0                                  # 亮度扰动也使 SSIM 下降，但幅度小
    assert s_lum > 0.9                                  # 加亮 0.05 后仍 > 0.9（亮度失真被宽容）
    assert s_struct < s_lum                             # 结构扰动降幅更大
    assert s_struct < 0.5                               # 翻转后结构项近 0，SSIM 大幅跌落


# ---------------------------------------------------------------------------
# 迁移等价性
# ---------------------------------------------------------------------------

def test_equivalence_with_original_inline_implementation():
    """与抽取前 run_nerf.py 内联实现逐位一致（< 1e-12）。

    _reference_psnr / _reference_ssim 是抽取前的内联实现原文（常数 1e-10、
    σ=1.5、k1=0.01、k2=0.03、L=1.0 逐字复制），在多组确定性图像对上比较，
    保证抽取重构没有改变任何数值语义。
    """
    rng = np.random.default_rng(2024)
    # 覆盖生产中真实出现的输入类型：float64 / float32（数据管线返回 float32）
    a64 = rng.random((24, 24, 3))
    b64 = np.clip(a64 + rng.normal(0.0, 0.1, a64.shape), 0.0, 1.0)
    a32 = (rng.random((20, 20, 3)) * 0.6 + 0.2).astype(np.float32)
    b32 = np.clip(a32 + rng.normal(0.0, 0.05, a32.shape), 0.0, 1.0).astype(np.float32)
    uniform_shift = a32 + 0.05                          # 亮度平移对
    flipped = a32[:, ::-1].copy()                       # 结构扰动对
    for a, b in [(a64, b64), (a32, b32), (a32, a32), (a32, uniform_shift), (a32, flipped)]:
        assert abs(psnr(a, b) - _reference_psnr(a, b)) < 1e-12
        assert abs(ssim(a, b) - _reference_ssim(a, b)) < 1e-12


if __name__ == "__main__":
    import traceback

    # 直接执行时：收集所有 test_ 开头的函数逐个运行，汇总通过率（不依赖 pytest）
    # 单点测试：python tests/test_metrics.py <测试名子串>；不带参数 = 全部测试。
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
