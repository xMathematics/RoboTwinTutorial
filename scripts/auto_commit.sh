#!/bin/bash

# 自动提交脚本
# 用法: ./scripts/auto_commit.sh
# 说明: 检查未跟踪/修改/暂存的变更，生成"新增文件/修改文件"清单式提交信息，
#       提交并推送到远程。遵守 .gitignore（papers/、logs/ 等不会被提交）。

set -e

echo "=== 自动提交脚本 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查是否在 git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "错误: 当前不在 git 仓库中"
    exit 1
fi

# 获取当前日期
TODAY=$(date '+%Y-%m-%d')

echo -e "${BLUE}1. 检查变更...${NC}"

# 检查是否有未跟踪的文件
UNTRACKED=$(git ls-files --others --exclude-standard | wc -l)
if [ "$UNTRACKED" -gt 0 ]; then
    echo "  - 未跟踪文件: $UNTRACKED 个"
fi

# 检查是否有修改的文件
MODIFIED=$(git diff --name-only | wc -l)
if [ "$MODIFIED" -gt 0 ]; then
    echo "  - 修改文件: $MODIFIED 个"
fi

# 检查是否有暂存的文件
STAGED=$(git diff --cached --name-only | wc -l)
if [ "$STAGED" -gt 0 ]; then
    echo "  - 暂存文件: $STAGED 个"
fi

# 检查是否有变更
if [ "$UNTRACKED" -eq 0 ] && [ "$MODIFIED" -eq 0 ] && [ "$STAGED" -eq 0 ]; then
    echo -e "${GREEN}没有需要提交的变更${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}2. 添加文件到暂存区...${NC}"

# 在 git add 之前先分类记录清单（add 之后全部变为"已暂存"，无法再区分新增/修改）
UNTRACKED_FILES=$(git ls-files --others --exclude-standard)
CHANGED_FILES=$(git diff --name-only HEAD)

# 添加所有变更文件
git add -A

echo -e "${GREEN}  ✓ 文件已添加${NC}"

echo ""
echo -e "${BLUE}3. 生成提交信息...${NC}"

# 生成提交信息：写入临时文件再以 git commit -F 提交
# （注意：不能用 "echo | while read" 管道拼接——管道右侧是子 shell，
#   循环内的变量赋值会随子 shell 退出而丢失；双引号内的 \n 也只是字面量）
COMMIT_FILE=$(mktemp)
trap 'rm -f "$COMMIT_FILE"' EXIT
{
    printf 'docs(global): 手动提交于 %s\n\n' "$TODAY"

    if [ -n "$UNTRACKED_FILES" ]; then
        printf '新增文件:\n'
        while IFS= read -r file; do
            printf '  - %s\n' "$file"
        done <<< "$UNTRACKED_FILES"
    fi

    if [ -n "$CHANGED_FILES" ]; then
        printf '\n修改文件:\n'
        while IFS= read -r file; do
            printf '  - %s\n' "$file"
        done <<< "$CHANGED_FILES"
    fi
} > "$COMMIT_FILE"

echo -e "${GREEN}  ✓ 提交信息已生成${NC}"
echo ""
echo -e "${BLUE}4. 执行提交...${NC}"

# 执行提交
git commit -F "$COMMIT_FILE"

echo -e "${GREEN}  ✓ 提交成功${NC}"

echo ""
echo -e "${BLUE}5. 推送到远程仓库...${NC}"

# 推送到远程仓库
git push

echo -e "${GREEN}  ✓ 推送成功${NC}"

echo ""
echo -e "${GREEN}=== 提交完成 ===${NC}"
