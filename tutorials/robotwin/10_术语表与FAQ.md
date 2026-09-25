# 第 10 章：术语表与常见问题（FAQ）

> 本章是全教程的"速查手册"。看不懂的缩写、想不起来的定义，都来这里查。

---

## 10.1 核心术语速查表（按字母序）

### A ~ C

| 术语 | 全称/中文 | 一句话解释 | 相关章节 |
|------|-----------|-----------|---------|
| **Action Chunking** | 动作分块 | 一次预测一段动作序列，减少累积误差（ACT 的核心） | 8 |
| **ACT** | Action Chunking with Transformers | 用 Transformer 一次预测多个动作的策略 | 8 |
| **Affordance** | 功能属性/可供性 | 物体"能做什么"的标注（可抓、可放、可按的位置） | 5 |
| **API** | 应用程序接口 | 论文中指机器人技能调用接口（grasp_actor 等） | 4 |
| **ASR** | Average Success Rate | 平均成功率（代码生成指标） | 6 |
| **Behavior Cloning** | 行为克隆 | 照着专家演示学做动作 | 1 |
| **Bimanual** | 双臂的 | 用两只机械臂协作完成操作 | 2 |
| **Benchmark** | 基准 | 统一的任务集与评测协议，用于比较算法 | 5 |
| **Code Agent** | 代码智能体 | 用大模型自动写机器人执行代码的模块 | 4 |
| **CodeBLEU / CodeBERT** | 代码相似度指标 | 衡量生成代码与人类代码的相似程度 | 6 |
| **CR-Iter** | 平均迭代轮数 | 代码生成达到 50% 成功率所需的平均反馈轮数 | 6 |
| **CuRobo** | NVIDIA 运动规划库 | GPU 加速的机器人运动规划器 | 2、7 |
| **Clutter** | 杂乱 | 桌面上加入无关干扰物（域随机化维度之一） | 4 |

### D ~ F

| 术语 | 全称/中文 | 一句话解释 | 相关章节 |
|------|-----------|-----------|---------|
| **Diffusion Policy (DP)** | 扩散策略 | 用扩散模型从噪声迭代生成动作 | 8 |
| **DP3** | 3D Diffusion Policy | 基于 3D 点云的扩散策略 | 8 |
| **Digital Twin** | 数字孪生 | 把真实世界 1:1 复刻进仿真 | 1 |
| **DoF** | Degrees of Freedom 自由度 | 机械臂可独立运动的方向数 | 2 |
| **Domain Randomization (DR)** | 域随机化 | 训练时随机化环境，增强 sim-to-real 鲁棒性 | 4 |
| **Embodiment** | 体态/平台 | 机器人的具体身体形态（Franka、UR5……） | 2 |
| **Embodiment-Aware** | 体态感知 | 根据机器人能力定制操作方案 | 4 |
| **Episode** | 回合/片段 | 从开始到结束的一次任务执行 | 2 |
| **End-to-End** | 端到端 | 从输入直接到输出，不经过中间人工模块 | 2 |
| **Epoch** | 轮次 | 把所有训练数据过一遍 | 8 |
| **Easy / Hard 设置** | 干净 / 随机化评测 | 评测环境的两种档位 | 6、8 |
| **Functional Point** | 功能点 | 物体上的功能位置标注（把手、按钮） | 5 |

### G ~ L

| 术语 | 全称/中文 | 一句话解释 | 相关章节 |
|------|-----------|-----------|---------|
| **Grasp Axis / Point** | 抓取轴/抓取点 | 物体上适合抓取的位置和方向 | 5 |
| **Gripper** | 夹爪 | 机械臂末端的抓取器 | 2 |
| **HDF5** | 分层数据格式 | 存储观测与动作的文件格式 | 2、7 |
| **Horizon** | 规划视野 | 预测未来多长一段动作 | 8 |
| **Imitation Learning** | 模仿学习 | 从演示中学习策略 | 1 |
| **Instruction** | 语言指令 | 告诉机器人任务目标的自然语言 | 4 |
| **Kinematics** | 运动学 | 描述机械臂关节与末端位置的数学关系 | 2 |
| **Leaderboard** | 排行榜 | 提交评测结果排名 | 8、9 |

### M ~ R

| 术语 | 全称/中文 | 一句话解释 | 相关章节 |
|------|-----------|-----------|---------|
| **MLLM** | Multimodal LLM 多模态大模型 | 能处理图像+文字的大模型 | 4 |
| **Motion Planner** | 运动规划器 | 规划从 A 到 B 的避障路径 | 2 |
| **Objaverse** | 开源 3D 资产库 | RoboTwin-OD 的物体来源之一 | 5 |
| **PartNet-Mobility** | 可活动物体数据集 | 提供铰接物体的来源 | 5 |
| **Pi0 (π0)** | VLA 流模型 | 通用机器人控制的基础模型 | 8 |
| **Policy** | 策略 | 机器人的"大脑"，观测→动作的映射 | 8 |
| **Pretrained** | 预训练的 | 已在大规模数据上训练过 | 8 |
| **RDT** | Robotic Diffusion Transformer | 双臂扩散基础模型 | 8 |
| **Reinforcement Learning** | 强化学习 | 通过试错+奖励学习 | 1 |
| **Rollout** | 试运行 | 让策略从头执行一次任务 | 6、8 |
| **RoboTwin-OD** | RoboTwin 对象数据集 | 731 个带标注的 3D 物体库 | 5 |

### S ~ Z

| 术语 | 全称/中文 | 一句话解释 | 相关章节 |
|------|-----------|-----------|---------|
| **SAPIEN** | 仿真器 | RoboTwin 2.0 的物理+渲染仿真平台 | 2 |
| **Sim-to-Real** | 仿真到现实迁移 | 把仿真里学的技能迁移到真实世界 | 1、6 |
| **Simulation-in-the-Loop** | 仿真内闭环 | 在仿真里执行并反馈修正 | 4 |
| **Stable Diffusion** | 文本生成图像模型 | 用来生成纹理库 | 4、5 |
| **Texture Library** | 纹理库 | 11000 张背景/桌面贴图 | 5 |
| **Token** | 词元 | 大模型处理文本的最小单位（成本指标） | 6 |
| **VLA** | Vision-Language-Action | 输入视觉+语言、输出动作的模型 | 8 |
| **VLM** | Vision-Language Model | 能看图理解的大模型（观察者角色） | 4 |
| **Zero-shot** | 零样本 | 不用该任务真实数据，直接泛化 | 6 |

---

## 10.2 常见问题 FAQ

### Q1：我没有机器人硬件，能学这篇论文吗？
**能**。RoboTwin 2.0 的所有数据生成与策略评测都可以在**纯仿真环境**中完成，只需一台 Linux + NVIDIA GPU 的电脑。真机部署只是论文实验的一部分，不是入门必需。

### Q2：没有 GPU 能不能学？
可以学**前 6 章**（概念与论文精读完全不需要 GPU）。第 7、8 章的实操需要 GPU；也可以借助云 GPU（如 AutoDL、Colab、阿里云）按小时租用。

### Q3：RoboTwin 2.0 和 RoboTwin 1.0 什么关系？
- **1.0**（CVPR 2025 Highlight）：用"数字孪生"镜像真实演示做双臂基准，14 个任务
- **2.0**（本文）：升级为"自动数据生成 + 强域随机化"，50 任务、5 平台、10 万+ 轨迹
- 简单说：2.0 = 1.0 的自动化 + 规模化 + 鲁棒化

### Q4：VLA、MLLM、VLM 有什么区别？
| 术语 | 输入 | 输出 | 例子 |
|------|------|------|------|
| **MLLM** | 图像+文字 | 文字/代码 | 代码生成智能体 |
| **VLM** | 图像+文字 | 文字理解/判断 | VLM 观察者（挑错） |
| **VLA** | 图像+文字 | **动作** | RDT、Pi0（机器人大脑） |

> 一句话：MLLM 会"看+写"；VLM 会"看+判断"；VLA 会"看+动手"。

### Q5：为什么要跑 10 次而不是 1 次？
仿真有随机性（动力学噪声、控制器抖动、传感器噪声）。跑 10 次取成功率，避免"运气好碰巧成功一次"造成误判。成功率 > 0.5 才收下代码。

### Q6：域随机化会不会让任务变得"太难学不会"？
有可能。论文实验也显示 Hard 条件下所有策略成功率下降。所以论文采取"剂量控制"：
- 干扰物排除与任务物体相似的（避免混淆）
- 光照/高度限制在物理合理范围
- 评测时才开 Hard 档，训练数据可以自由选择是否随机化

### Q7：为什么论文说"干净数据预训练没用"？
实验三显示：只用干净数据预训练，在随机化环境下成功率不升反降。因为干净数据里**没有"环境变化"这个概念**，模型没学会"无视干扰"。而随机化数据教会了模型"环境怎么变都能应对"，这个能力保存在预训练权重里。

### Q8：我该用哪个策略入门？
建议顺序：**ACT → DP → DP3 → RDT/Pi0**。
- ACT/DP：训练快、资源少，适合跑通流程
- DP3：适合点云、样本高效，但依赖完美点云
- RDT/Pi0：预训练 VLA，最贴近前沿，但资源需求高

### Q9：采集数据时卡住怎么办？
- 避免 A/H/V 系列 GPU（官方 issue #83）
- Docker 环境确保 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`
- 检查 Vulkan：`vulkaninfo`
- 查看官方 common-issue 页面

### Q10：生成的代码成功率为什么不是 100%？
论文附录显示平均代码生成成功率约 43.34%。原因：
- 不同任务难度差异大（简单的 100%，难的如 click_bell 仅 10%）
- 依赖 MLLM 能力与 API 设计
- VLM 观察者自身也有误报（Precision 0.208）
这正是该领域的重要研究空间。

### Q11：这篇论文的代码和数据集在哪？
| 资源 | 地址 |
|------|------|
| 代码仓库 | https://github.com/RoboTwin-Platform/RoboTwin |
| 官方文档 | https://robotwin-platform.github.io/doc/ |
| 项目主页 | https://robotwin-platform.github.io/ |
| 数据集 | https://huggingface.co/datasets/TianxingChen/RoboTwin2.0 |
| Leaderboard | https://robotwin-platform.github.io/leaderboard |

### Q12：如何引用这篇论文？
```bibtex
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization
         for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin
          and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

---

## 10.3 全教程"一页速查"（建议打印）

```text
【数据生成闭环】任务描述 + API库 → MLLM写代码 → 仿真跑10次 → (日志+VLM挑错) → 修复 → 成功率>50%收下
【五维域随机化】杂乱 / 光照 / 背景纹理 / 桌面高度 / 语言指令
【体态感知抓取】物体多轴标注 → 可达性偏向扰动 → 并行规划验证 → 机器人专属抓法
【对象库】RoboTwin-OD：731物体 / 147类，每物体15条语言描述 + 放置/功能/抓取点轴
【基准】50任务 × 5平台(Aloha/ARX/Piper/Franka/UR5)，10万+轨迹
【评测协议】Easy(干净) vs Hard(随机化)，每任务50条训练 + 100次rollout
【关键结论】体态抓取提升低DoF(Piper 2.4%→25.1%)；随机化预训练RDT+31.9%；10真机+1k合成真机提升24.4%
```

---

## 10.4 结语

恭喜你完成整套零基础教程！现在你已经：
1. 理解了具身智能、双臂操作、仿真的基本概念
2. 能逐节精读一篇顶会论文并批判性思考
3. 掌握了三大核心技术（代码生成闭环、域随机化、体态感知抓取）的原理
4. 知道如何安装环境、采集数据、训练评测策略
5. 找到了进一步学习和研究的方向

**下一步建议**：打开第 9 章的学习路线，动手跑通第一个任务。纸上得来终觉浅，绝知此事要躬行。

> 🎓 *祝你学习顺利，期待未来在具身智能的赛道上看到你的成果！*
