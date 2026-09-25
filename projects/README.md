# projects/ — 代码项目

可运行的参考实现，按主题分子目录。

| 主题 | 说明 | 入口 |
|------|------|------|
| [nerf/](nerf/) | NeRF PyTorch 教学版参考实现（位置编码 / coarse-fine 双 MLP / 分层采样 / 可微体积渲染） | [README.md](nerf/README.md) |

## nerf/ 快速开始

```bash
conda activate llm_env
cd projects/nerf

# 单元测试（8/8）
python tests/test_core.py

# 生成演示数据 → 冒烟训练 → 评估渲染
python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32
python run_nerf.py --config data/demo_scene --mode train --exp demo_smoke --tiny --steps 80
python run_nerf.py --config data/demo_scene --mode test --exp demo_smoke --ckpt logs/demo_smoke/latest.pt
```

VS Code 中可直接用 Run and Debug 或任务面板执行（配置见 `.vscode/`）。

## 目录约定

- 训练产物（`logs/`、`data/`）不入库，已在根 `.gitignore` 忽略
- 代码遵循 PEP 8 与类型提示（见 [CONSTRAINTS.md](../CONSTRAINTS.md) §4）
- 教学简化处会在项目 README 中注明；完整复现以论文官方实现为准
