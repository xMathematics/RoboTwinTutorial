"""NeRF：把场景表示为神经辐射场（Neural Radiance Fields）并合成新视角。

Mildenhall et al. (ECCV 2020) 的极简教学版 PyTorch 复现。支持 Blender 合成数据集
（nerf_synthetic），覆盖全部核心机制：位置编码（positional encoding）、coarse/fine
双 MLP、分层采样（stratified sampling）与层次采样（hierarchical sampling）、可微
体积渲染（volume rendering）、coarse+fine 联合训练目标。

函数流水线（本包在 NeRF 训练/渲染管线中的整体结构，箭头表示"调用"）：

    run_nerf.py（命令行入口）
      ├→ config.Config                    超参数（无包内依赖）
      ├→ data_utils.load_blender_data     读图像+位姿 → 光线网格（依赖 rays.get_rays_np）
      ├→ rays.get_rays                    由 c2w 位姿生成光线（test/render 模式用）
      └→ trainer.train_nerf               训练循环（依赖 config、model、render）
           └→ render.render_rays          对一批光线做 coarse + fine 两趟渲染
                ├→ sampling.stratified_sample / hierarchical_sample / ray_deltas
                ├→ encoding.positional_encoding
                ├→ model.NeRF             coarse、fine 两个同构独立 MLP
                └→ volume_render          离散体积渲染（论文 Eq. 3）

输入：nerf_synthetic 场景目录（transforms_*.json + 图像）；输出：coarse/fine 权重
checkpoint、渲染图像与 PSNR/SSIM 指标。
"""

__version__ = "0.1.0"
