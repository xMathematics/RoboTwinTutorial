# 自动化脚本说明

本目录包含用于自动化的脚本。

## auto_commit.sh

自动提交代码到 Git 仓库的脚本。

### 功能

- 检查项目中的变更
- 自动添加文件到暂存区
- 生成规范的提交信息
- 提交变更到本地仓库
- 推送到远程仓库

### 使用方法

```bash
# 在项目根目录执行
./scripts/auto_commit.sh
```

### 工作流程

1. 检查变更：查看未跟踪、修改和暂存的文件
2. 添加文件：将所有变更文件添加到暂存区
3. 生成提交信息：自动生成包含日期和变更内容的提交信息
4. 执行提交：提交到本地仓库
5. 推送变更：推送到远程仓库

### 提交信息格式

```
docs(global): 手动提交于 YYYY-MM-DD

新增文件:
  - file1.md
  - file2.py

修改文件:
  - file3.md
```

### 注意事项

- 确保已配置好 Git 远程仓库
- 确保已配置好 Git 凭据
- 脚本会在推送失败时停止执行

### 手动推送

如果只想提交但不推送，可以：

```bash
# 只提交，不推送
git add -A
git commit -m "your commit message"
git push  # 如果想推送
```

## GitHub Actions

项目还配置了 GitHub Actions 工作流，自动每周一上午 9 点提交代码。

### 配置文件

- `.github/workflows/auto-commit.yml` - GitHub Actions 工作流配置

### 手动触发

在 GitHub 仓库页面：
1. 进入 Actions 标签
2. 选择 "定期提交代码" 工作流
3. 点击 "Run workflow" 按钮
4. 选择分支并运行

### 自定义提交频率

编辑 `.github/workflows/auto-commit.yml` 文件中的 cron 表达式：

```yaml
schedule:
  - cron: '0 9 * * 1'  # 每周一上午 9 点
```

cron 表达式格式：`分 时 日 月 周`

常用示例：
- `0 9 * * 1` - 每周一上午 9 点
- `0 9 * * *` - 每天上午 9 点
- `0 9 * * 1-5` - 每周一到周五上午 9 点
- `0 9 1 * *` - 每月 1 号上午 9 点
