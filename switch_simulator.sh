#!/bin/bash

# Simulator Switcher Script
# 一键切换 IsaacGym, Genesis, IsaacLab 仿真器环境

# 配置
CONDA_ENV_GYM="lr_gym"
CONDA_ENV_GEN="lr_gen"
CONDA_ENV_LAB="lr_lab"

detect_conda_root() {
    local conda_base
    if command -v conda &> /dev/null; then
        conda_base=$(conda info --base 2>/dev/null)
        if [ -n "$conda_base" ]; then
            echo "$conda_base"
            return 0
        fi
    fi
    
    local user_home="$HOME"
    if [ -d "$user_home/miniconda3" ]; then
        echo "$user_home/miniconda3"
    elif [ -d "$user_home/anaconda3" ]; then
        echo "$user_home/anaconda3"
    elif [ -d "/opt/miniconda3" ]; then
        echo "/opt/miniconda3"
    elif [ -d "/opt/anaconda3" ]; then
        echo "/opt/anaconda3"
    else
        echo ""
    fi
}

CONDA_ROOT=$(detect_conda_root)
if [ -z "$CONDA_ROOT" ]; then
    echo "警告: 无法自动检测conda安装目录"
    echo "将尝试使用默认路径: $HOME/miniconda3"
    CONDA_ROOT="$HOME/miniconda3"
fi

CONDA_DIR_NAME=$(basename "$CONDA_ROOT")
USER_NAME=$(whoami)
USER_HOME="$HOME"

echo "=========================================="
echo "       Conda环境检测信息"
echo "=========================================="
echo "当前用户名: $USER_NAME"
echo "用户主目录: $USER_HOME"
echo "Conda根目录: $CONDA_ROOT"
echo "Conda目录名: $CONDA_DIR_NAME"
echo "=========================================="
echo ""

# 检测conda环境是否存在
check_conda_env() {
    local env_name=$1
    conda env list | grep -q "^${env_name}\s"
    return $?
}

# 主函数
main() {
    echo "=========================================="
    echo "       Simulator Switcher Tool"
    echo "=========================================="
    echo ""
    
    # 检测可用的环境
    echo "检测conda环境..."
    local has_gym=false
    local has_gen=false
    local has_lab=false
    
    if check_conda_env "$CONDA_ENV_GYM"; then
        echo "  [✓] $CONDA_ENV_GYM (IsaacGym)"
        has_gym=true
    else
        echo "  [✗] $CONDA_ENV_GYM (IsaacGym) - 不可用"
    fi
    
    if check_conda_env "$CONDA_ENV_GEN"; then
        echo "  [✓] $CONDA_ENV_GEN (Genesis)"
        has_gen=true
    else
        echo "  [✗] $CONDA_ENV_GEN (Genesis) - 不可用"
    fi
    
    if check_conda_env "$CONDA_ENV_LAB"; then
        echo "  [✓] $CONDA_ENV_LAB (IsaacLab)"
        has_lab=true
    else
        echo "  [✗] $CONDA_ENV_LAB (IsaacLab) - 不可用"
    fi
    
    echo ""
    echo "可用选项: isaacgym, genesis, isaaclab"
    echo "请输入要切换的simulator (或输入 'exit' 退出):"
    read -r user_input
    
    # 转换为小写
    user_input=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')
    
    case "$user_input" in
        "isaacgym")
            if [ "$has_gym" = false ]; then
                echo "错误: Conda环境 '$CONDA_ENV_GYM' 不存在！"
                echo "请确保已创建该环境。"
                return 1
            fi
            echo "切换到 IsaacGym..."
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV_GYM"
            export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV_GYM}/lib:${LD_LIBRARY_PATH}"
            echo "成功切换到 IsaacGym 环境!"
            echo "激活环境: $CONDA_ENV_GYM"
            echo "设置 LD_LIBRARY_PATH: ${CONDA_ROOT}/envs/${CONDA_ENV_GYM}/lib"
            ;;
            
        "genesis")
            if [ "$has_gen" = false ]; then
                echo "错误: Conda环境 '$CONDA_ENV_GEN' 不存在！"
                echo "请确保已创建该环境。"
                return 1
            fi
            echo "切换到 Genesis..."
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV_GEN"
            export SIMULATOR=genesis
            echo "成功切换到 Genesis 环境!"
            echo "激活环境: $CONDA_ENV_GEN"
            echo "设置 SIMULATOR=genesis"
            ;;
            
        "isaaclab")
            if [ "$has_lab" = false ]; then
                echo "错误: Conda环境 '$CONDA_ENV_LAB' 不存在！"
                echo "请确保已创建该环境。"
                return 1
            fi
            echo "切换到 IsaacLab..."
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV_LAB"
            export SIMULATOR=isaaclab
            echo "成功切换到 IsaacLab 环境!"
            echo "激活环境: $CONDA_ENV_LAB"
            echo "设置 SIMULATOR=isaaclab"
            ;;
            
        "exit"|"quit"|"q")
            echo "退出脚本。"
            return 0
            ;;
            
        *)
            echo "无效输入: '$user_input'"
            echo "请输入以下选项之一: isaacgym, genesis, isaaclab"
            return 1
            ;;
    esac
}

# 执行主函数
main
