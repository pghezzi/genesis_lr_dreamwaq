#!/bin/bash

# Simulator Switcher Script
# 一键切换 IsaacGym, Genesis, IsaacLab 仿真器环境

# 配置
CONDA_ENV_GYM="lr_gym"
CONDA_ENV_GEN="lr_gen"
CONDA_ENV_LAB="lr_lab"

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
            export LD_LIBRARY_PATH=/home/lupinjia/miniconda3/envs/lr_gym/lib:$LD_LIBRARY_PATH
            echo "成功切换到 IsaacGym 环境!"
            echo "激活环境: $CONDA_ENV_GYM"
            echo "设置 LD_LIBRARY_PATH: /home/lupinjia/miniconda3/envs/lr_gym/lib"
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
