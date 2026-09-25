# 变更日志

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增
- 全局约束文档 (CONSTRAINTS.md)
- 变更日志 (CHANGELOG.md)
- 文档规范新增 §3.4「论述结构规范（问题导向五步法）」：核心知识点必须按"问题场景 → 解决方法 → 选型理由 → 理论依据 → 完整推导"展开，推导禁止跳步；附 NeRF 体积渲染完整示范与检查清单项
- SLAM 教程架构文档：10 章规划，每章预埋问题场景锚点与推导产出清单，含学习路径依赖图与资源清单

### 变更
- **项目重构**：按资源类型重新划分目录——`papers/`（论文库）、`tutorials/`（教程文档）、`projects/`（代码项目），原 `nerf/`、`robotwin/`、`slam/`、`3d_reconstruction/` 按主题拆分归入三类
- 同步更新根 README、各索引 README、`.vscode/` 配置、`.gitignore` 及 CONSTRAINTS.md

## [1.0.0] - 2026-09-25

### 新增
- NeRF 模块基础教程
- RoboTwin2.0 模块教程
- SLAM 模块教程
- 3D 重建模块教程
- 项目全局文档结构

### 文档
- 各模块 README
- 各模块教程文档
- 全局约束规范

---

## 版本说明

### [Unreleased] - 开发中
当前开发版本，包含未发布的新功能和变更。

### [1.0.0] - 初始版本
项目初始发布版本，包含所有基础模块和文档。

---

## 变更类型

- **新增** (Added): 新功能
- **变更** (Changed): 现有功能的变更
- **弃用** (Deprecated): 即将移除的功能
- **移除** (Removed): 已移除的功能
- **修复** (Fixed): Bug 修复
- **安全** (Security): 安全性相关修复

## 变更追踪

- **问题**: [链接到 GitHub Issue]
- **PR**: [链接到 Pull Request]
