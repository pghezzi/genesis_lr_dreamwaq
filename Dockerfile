# =============================================================================
# LeggedGym-Ex Dockerfile (Multi-Simulator)
# =============================================================================
# 支持三个模拟器：IsaacGym、Genesis、IsaacSim
#
# 构建命令：
#   cd /path/to/LeggedGym-Ex
#   docker build -t leggedgym-ex:all -f Dockerfile ..
#
# 注意：构建上下文需要是上级目录，因为需要COPY isaacgym
#
# 运行命令：
#   # Genesis 环境
#   docker run --gpus all -it --rm leggedgym-ex:all genesis
#
#   # IsaacGym 环境
#   docker run --gpus all -it --rm leggedgym-ex:all isaacgym
#
#   # IsaacSim 环境
#   docker run --gpus all -it --rm leggedgym-ex:all isaacsim

# ---- 基础镜像 ----
# 使用CUDA 12.8（IsaacSim需要），PyTorch可向下兼容cu121/cu126
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# ---- 复制 uv 工具 ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- 环境变量 ----
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_HTTP_TIMEOUT=300

# ---- 安装系统依赖 ----
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libegl-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libvulkan-dev \
    vulkan-tools \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-pip \
    bash-completion \
    vim \
    ncurses-bin \
    && rm -rf /var/lib/apt/lists/*

# ---- 配置 PyPI 镜像源（加速下载）----
RUN python3 -m pip config set global.index-url \
    https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 安装三个Python版本 ----
RUN uv python install 3.8 3.10 3.11

# ---- 创建工作目录 ----
WORKDIR /workspace

# ---- 复制 IsaacGym ----
COPY isaacgym /workspace/isaacgym

# ---- 复制 IsaacLab（预先下载）----
COPY IsaacLab /workspace/IsaacLab

# ---- 复制项目文件 ----
COPY LeggedGym-Ex/pyproject.toml /workspace/LeggedGym-Ex/
COPY LeggedGym-Ex/rsl_rl /workspace/LeggedGym-Ex/rsl_rl
COPY LeggedGym-Ex/legged_gym /workspace/LeggedGym-Ex/legged_gym
COPY LeggedGym-Ex/resources /workspace/LeggedGym-Ex/resources

# ---- 创建并安装 IsaacGym 环境 (Python 3.8 + cu121) ----
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /workspace/LeggedGym-Ex && \
    uv venv --python 3.8 .venv-isaacgym && \
    . .venv-isaacgym/bin/activate && \
    uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    cd /workspace/isaacgym/python && \
    find . -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} + && \
    uv pip install -e . && \
    cd /workspace/LeggedGym-Ex && \
    uv pip install ".[isaacgym]"

# ---- 创建并安装 Genesis 环境 (Python 3.10 + cu126) ----
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /workspace/LeggedGym-Ex && \
    uv venv --python 3.10 .venv-genesis && \
    . .venv-genesis/bin/activate && \
    uv pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    uv pip install ".[genesis]"

# ---- 创建并安装 IsaacSim 环境 (Python 3.11 + cu128) ----
ENV TERM=xterm
RUN touch /.dockerenv
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /workspace/LeggedGym-Ex && \
    uv venv --python 3.11 .venv-isaaclab && \
    . .venv-isaaclab/bin/activate && \
    uv pip install "isaacsim[all,extscache]==5.1.0" \
        --extra-index-url https://pypi.nvidia.com && \
    cd /workspace/IsaacLab && \
    ./isaaclab.sh --install none && \
    cd /workspace/LeggedGym-Ex && \
    uv pip install matplotlib tensorboard xlsxwriter pandas wandb tqdm scipy numpy pygame trimesh rich-argparse && \
    uv pip install warp-lang==1.12.0 && \
    uv pip install -e . --no-deps

# ---- 复制项目源码 ----
COPY LeggedGym-Ex /workspace/LeggedGym-Ex

# ---- 重新安装项目（editable mode）----
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /workspace/LeggedGym-Ex && \
    . .venv-isaacgym/bin/activate && uv pip install -e . && \
    . .venv-genesis/bin/activate && uv pip install -e . && \
    . .venv-isaaclab/bin/activate && uv pip install -e .

# ---- 创建启动脚本 ----
RUN cat > /workspace/entrypoint.sh << 'EOF'
#!/bin/bash
case "$1" in
    isaacgym)
        source /workspace/LeggedGym-Ex/.venv-isaacgym/bin/activate
        export SIMULATOR=isaacgym
        export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
        ;;
    genesis)
        source /workspace/LeggedGym-Ex/.venv-genesis/bin/activate
        export SIMULATOR=genesis
        ;;
    isaacsim|isaaclab)
        source /workspace/LeggedGym-Ex/.venv-isaaclab/bin/activate
        export SIMULATOR=isaaclab
        ;;
    *)
        echo "Usage: docker run ... leggedgym-ex:all [isaacgym|genesis|isaacsim] [command]"
        echo ""
        echo "Available simulators:"
        echo "  isaacgym  - IsaacGym (Python 3.8, cu121)"
        echo "  genesis   - Genesis (Python 3.10, cu126)"
        echo "  isaacsim  - IsaacSim/IsaacLab (Python 3.11, cu128)"
        echo ""
        echo "Example:"
        echo "  docker run --gpus all -it --rm leggedgym-ex:all genesis bash"
        exit 1
        ;;
esac

cd /workspace/LeggedGym-Ex
export PYTHONPATH=/workspace/LeggedGym-Ex:$PYTHONPATH

if [ $# -gt 1 ]; then
    exec "${@:2}"
else
    exec bash
fi
EOF
RUN chmod +x /workspace/entrypoint.sh

# ---- 默认入口 ----
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["genesis", "bash"]
