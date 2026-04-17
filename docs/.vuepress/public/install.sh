#!/bin/bash
# ─────────────────────────────────────────────────────────────
# DMLA Sandbox 安装脚本
# 一键安装启动器，用于检测环境并调用 TUI 安装向导
# ─────────────────────────────────────────────────────────────

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 打印欢迎信息
echo ""
echo "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo "${BOLD}${BLUE}║                                                            ║${NC}"
echo "${BOLD}${BLUE}║           DMLA Sandbox 安装向导                            ║${NC}"
echo "${BOLD}${BLUE}║                                                            ║${NC}"
echo "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ─────────────────────────────────────────────────────────────
# 检查 Docker
# ─────────────────────────────────────────────────────────────
echo "${BOLD}🔍 环境检测${NC}"
echo ""

if ! command -v docker &> /dev/null; then
    echo "${RED}❌ Docker 未安装${NC}"
    echo ""
    echo "${YELLOW}💡 请先安装 Docker:${NC}"
    echo "   macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "   Linux: https://docs.docker.com/engine/install/"
    echo "   Windows: https://docs.docker.com/desktop/install/windows-install/"
    echo ""
    exit 1
fi

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo "${RED}❌ Docker 未运行${NC}"
    echo ""
    echo "${YELLOW}💡 请启动 Docker 服务:${NC}"
    echo "   macOS: 打开 Docker Desktop"
    echo "   Linux: sudo systemctl start docker"
    echo ""
    exit 1
fi

DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "未知")
echo "${GREEN}✅ Docker $DOCKER_VERSION 已安装并运行${NC}"

# ─────────────────────────────────────────────────────────────
# 检查 Node.js
# ─────────────────────────────────────────────────────────────
if ! command -v node &> /dev/null; then
    echo "${RED}❌ Node.js 未安装${NC}"
    echo ""
    echo "${YELLOW}💡 请先安装 Node.js:${NC}"
    echo "   推荐版本: Node.js 18+ 或 20+"
    echo "   下载地址: https://nodejs.org/"
    echo ""
    echo "${YELLOW}安装 Node.js 后，请重新运行此脚本:${NC}"
    echo "   curl -fsSL https://ai.icyfenix.cn/install.sh | sh"
    echo ""
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/v//')
echo "${GREEN}✅ Node.js $NODE_VERSION 已安装${NC}"

# ─────────────────────────────────────────────────────────────
# 检查 GPU (可选)
# ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi -L &> /dev/null; then
        GPU_INFO=$(nvidia-smi -L | head -1 | sed 's/GPU [0-9]*: //' | sed 's/ (UUID: .*)//')
        echo "${GREEN}✅ GPU: $GPU_INFO${NC}"
    else
        echo "${NC}   GPU: 未检测到 (可选)${NC}"
    fi
else
    echo "${NC}   GPU: 未检测到 (可选)${NC}"
fi

echo ""

# ─────────────────────────────────────────────────────────────
# 检查 npx 是否可用
# ─────────────────────────────────────────────────────────────
if ! command -v npx &> /dev/null; then
    echo "${RED}❌ npx 不可用${NC}"
    echo "${YELLOW}💡 npx 通常随 Node.js 安装，请检查 Node.js 安装是否完整${NC}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# 启动 TUI 安装向导
# ─────────────────────────────────────────────────────────────
echo "${BOLD}📦 启动安装向导...${NC}"
echo ""

# 使用 npx 运行 @dmla/install
# --yes 参数自动接受 npx 的安装确认
npx --yes @dmla/install

# ─────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}${GREEN}🎉 安装完成！${NC}"
echo ""
echo "${NC}常用命令:${NC}"
echo "  dmla start      启动服务"
echo "  dmla status     查看状态"
echo "  dmla update     更新版本"
echo "  dmla doctor     环境诊断"
echo ""