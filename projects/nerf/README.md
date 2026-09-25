# NeRF 最小参考实现（教学版）

这是论文 *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*
（Mildenhall et al., ECCV 2020）的**最小可运行 PyTorch 复现**，用于配合
`../tutorial/` 的教程逐行学习。代码刻意保持精简，只保留论文的核心机制。

## 功能

- ✅ 位置编码 `γ(p)`（论文 Eq.4）
- ✅ 双分支 MLP（密度 / 视角相关颜色，含 skip connection，论文 Fig.7）
- ✅ 分层采样（stratified sampling，论文 Eq.2）
- ✅ 层次采样（hierarchical sampling，论文 Sec.5.2）
- ✅ 可微体积渲染 / alpha 合成（论文 Eq.3）
- ✅ coarse + fine 双网络联合训练（论文 Eq.7）
- ✅ 训练 / 测试（PSNR/SSIM）/ 渲染相机轨迹

## 环境

使用 **conda**（推荐 `llm_env`，已含 torch 2.11.0；也可新建专属环境）：

```bash
conda activate llm_env
pip install -r requirements.txt     # 缺什么装什么
```

> VS Code 用户：`.vscode/settings.json` 已将 Python 解释器指向 `llm_env`，
> 并提供调试配置（`launch.json`）与任务（`tasks.json`）。

依赖：`torch>=2.0`、`numpy`、`imageio`、`scipy`、`tqdm`、`Pillow`。

## 数据

**快速体验**：无需下载，用脚本生成演示数据集（彩色小球，多视角一致）：

```bash
python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32
```

**官方数据**：支持 **Blender 合成数据集（nerf_synthetic）** 标准格式：

```
data/lego/
├── transforms_train.json / val.json / test.json
└── train/ val/ test/  *.png
```

官方下载（约 150MB/场景）：
https://drive.google.com/drive/folders/128yBriW1IG_3NH5nEukoXhCF7d8Ttzl3

## 用法

```bash
# 冒烟测试（CPU，~1 分钟）验证代码正确性
python tests/test_core.py
# 或 pytest

# 用演示数据快速训练（80 步，~5 秒 CPU）
python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32
python run_nerf.py --config data/demo_scene --mode train --exp demo_smoke --tiny --steps 80

# 训练（V100 全量约 6~10h；tiny 模式 ~5min 看损失下降）
python run_nerf.py --config data/lego --mode train --exp lego_full --steps 200000
python run_nerf.py --config data/lego --mode train --exp lego_smoke --tiny

# 测试（渲染测试集新视角 + PSNR/SSIM）
python run_nerf.py --config data/lego --mode test --exp lego_full --ckpt logs/lego_full/latest.pt

# 渲染相机轨迹视频帧
python run_nerf.py --config data/lego --mode render --exp lego_full \
    --ckpt logs/lego_full/latest.pt --frames 120
```

## 代码结构

| 文件 | 对应论文 | 说明 |
|------|---------|------|
| `nerf/encoding.py` | Eq.4 | 位置编码 |
| `nerf/model.py` | Sec.3 / Fig.7 | NeRF MLP |
| `nerf/sampling.py` | Eq.2 / Sec.5.2 | 分层 + 层次采样 |
| `nerf/render.py` | Eq.3 / Sec.5.2 | 体积渲染 + coarse/fine 渲染管线 |
| `nerf/rays.py` | — | 相机→光线（Blender 约定） |
| `nerf/data_utils.py` | Sec.6.1 | 加载 nerf_synthetic |
| `nerf/trainer.py` | Eq.7 / Sec.5.3 | 训练循环 |
| `run_nerf.py` | — | 命令行入口 |

> ⚠️ 本实现为**教学简化版**：只支持 Blender 合成数据、未实现 NDC/COLMAP、
> 未做多分辨率渲染等。追求完整复现请参考官方实现
> [bmild/nerf](https://github.com/bmild/nerf)（TensorFlow）。
