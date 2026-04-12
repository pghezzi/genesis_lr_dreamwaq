#!/bin/bash

# Simulator Switcher Script
# One-click switch between IsaacGym, Genesis, IsaacLab simulators

# Color definitions
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
MAGENTA='\033[35m'
CYAN='\033[36m'
RESET='\033[0m'

# Configuration
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
    echo -e "${YELLOW}Warning: Unable to auto-detect conda installation directory${RESET}"
    echo -e "${YELLOW}Will use default path: $HOME/miniconda3${RESET}"
    CONDA_ROOT="$HOME/miniconda3"
fi

CONDA_DIR_NAME=$(basename "$CONDA_ROOT")
USER_NAME=$(whoami)
USER_HOME="$HOME"

echo -e "${BLUE}==========================================${RESET}"
echo -e "${BLUE}       Conda Environment Detection${RESET}"
echo -e "${BLUE}==========================================${RESET}"
echo -e "${CYAN}Current User:${RESET} $USER_NAME"
echo -e "${CYAN}User Home:${RESET} $USER_HOME"
echo -e "${CYAN}Conda Root:${RESET} $CONDA_ROOT"
echo -e "${CYAN}Conda Directory:${RESET} $CONDA_DIR_NAME"
echo -e "${BLUE}==========================================${RESET}"
echo ""

# Check if conda environment exists
check_conda_env() {
    local env_name=$1
    conda env list | grep -q "^${env_name}\s"
    return $?
}

# Main function
main() {
    echo -e "${BLUE}==========================================${RESET}"
    echo -e "${BLUE}       Simulator Switcher Tool${RESET}"
    echo -e "${BLUE}==========================================${RESET}"
    echo ""
    
    # Check available environments
    echo -e "${CYAN}Detecting conda environments...${RESET}"
    local has_gym=false
    local has_gen=false
    local has_lab=false
    
    if check_conda_env "$CONDA_ENV_GYM"; then
        echo -e "  ${GREEN}[✓]${RESET} $CONDA_ENV_GYM (IsaacGym)"
        has_gym=true
    else
        echo -e "  ${RED}[✗]${RESET} $CONDA_ENV_GYM (IsaacGym) - ${YELLOW}Not Available${RESET}"
    fi
    
    if check_conda_env "$CONDA_ENV_GEN"; then
        echo -e "  ${GREEN}[✓]${RESET} $CONDA_ENV_GEN (Genesis)"
        has_gen=true
    else
        echo -e "  ${RED}[✗]${RESET} $CONDA_ENV_GEN (Genesis) - ${YELLOW}Not Available${RESET}"
    fi
    
    if check_conda_env "$CONDA_ENV_LAB"; then
        echo -e "  ${GREEN}[✓]${RESET} $CONDA_ENV_LAB (IsaacLab)"
        has_lab=true
    else
        echo -e "  ${RED}[✗]${RESET} $CONDA_ENV_LAB (IsaacLab) - ${YELLOW}Not Available${RESET}"
    fi
    
    echo ""
    echo -e "${CYAN}Available Options:${RESET} isaacgym, genesis, isaaclab"
    echo -e "${MAGENTA}Enter the simulator to switch (or type 'exit' to quit):${RESET}"
    read -r user_input
    
    # Convert to lowercase
    user_input=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')
    
    case "$user_input" in
        "isaacgym")
            if [ "$has_gym" = false ]; then
                echo -e "${RED}Error: Conda environment '$CONDA_ENV_GYM' does not exist!${RESET}"
                echo -e "${YELLOW}Please ensure this environment is created.${RESET}"
                return 1
            fi
            echo -e "${CYAN}Switching to IsaacGym...${RESET}"
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV_GYM"
            export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${CONDA_ENV_GYM}/lib:${LD_LIBRARY_PATH}"
            echo -e "${GREEN}Successfully switched to IsaacGym environment!${RESET}"
            echo -e "${CYAN}Activated Environment:${RESET} $CONDA_ENV_GYM"
            echo -e "${CYAN}Set LD_LIBRARY_PATH:${RESET} ${CONDA_ROOT}/envs/${CONDA_ENV_GYM}/lib"
            ;;
            
        "genesis")
            if [ "$has_gen" = false ]; then
                echo -e "${RED}Error: Conda environment '$CONDA_ENV_GEN' does not exist!${RESET}"
                echo -e "${YELLOW}Please ensure this environment is created.${RESET}"
                return 1
            fi
            echo -e "${CYAN}Switching to Genesis...${RESET}"
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV_GEN"
            export SIMULATOR=genesis
            echo -e "${GREEN}Successfully switched to Genesis environment!${RESET}"
            echo -e "${CYAN}Activated Environment:${RESET} $CONDA_ENV_GEN"
            echo -e "${CYAN}Set SIMULATOR=genesis${RESET}"
            ;;
            
        "isaaclab")
            if [ "$has_lab" = false ]; then
                echo -e "${RED}Error: Conda environment '$CONDA_ENV_LAB' does not exist!${RESET}"
                echo -e "${YELLOW}Please ensure this environment is created.${RESET}"
                return 1
            fi
            echo -e "${CYAN}Switching to IsaacLab...${RESET}"
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV_LAB"
            export SIMULATOR=isaaclab
            echo -e "${GREEN}Successfully switched to IsaacLab environment!${RESET}"
            echo -e "${CYAN}Activated Environment:${RESET} $CONDA_ENV_LAB"
            echo -e "${CYAN}Set SIMULATOR=isaaclab${RESET}"
            ;;
            
        "exit"|"quit"|"q")
            echo -e "${YELLOW}Exiting script.${RESET}"
            return 0
            ;;
            
        *)
            echo -e "${RED}Invalid input: '$user_input'${RESET}"
            echo -e "${YELLOW}Please enter one of the following: isaacgym, genesis, isaaclab${RESET}"
            return 1
            ;;
    esac
}

# Execute main function
main
