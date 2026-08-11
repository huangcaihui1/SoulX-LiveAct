#!/bin/bash
# ==============================================================================
# SoulX-LiveAct 环境初始化脚本
# 适用于 AutoDL + RTX PRO 6000 (Blackwell) + PyTorch 2.8.0 + CUDA 12.8
#
# 使用方法:
#   bash setup.sh env      # 步骤 1-5: 创建 conda 环境 + 安装 Python 依赖
#   bash setup.sh kernel   # 步骤 6-7: 安装 LightX2V + lightx2v_kernel (FP4)
#   bash setup.sh models   # 步骤 8:  下载模型权重
#   bash setup.sh verify   # 步骤 9:  验证安装
#   bash setup.sh all      # 全部执行
# ==============================================================================

set -e

# ==================== 配置 ====================
CONDA_ENV_NAME="liveact"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"
TMP_DIR="/tmp/liveact_build"

# GitHub 镜像加速 (国内访问 github 慢时使用)
# 优先级: GHPROXY > GHGO > 直接访问
GITHUB_MIRROR="${GITHUB_MIRROR:-https://ghproxy.com}"

# 包装 git clone, 自动使用镜像
git_clone() {
    local url="$1"
    local target="$2"
    local depth="${3:-}"

    # 先尝试直接 clone (如果网络好)
    if [ -n "$depth" ]; then
        if git clone --depth ${depth} "${url}" "${target}" 2>/dev/null; then
            return 0
        fi
    else
        if git clone "${url}" "${target}" 2>/dev/null; then
            return 0
        fi
    fi

    # 直接失败, 使用镜像
    local mirrored_url="${GITHUB_MIRROR}/${url}"
    log_warn "直接 clone 失败, 尝试镜像: ${mirrored_url}"
    if [ -n "$depth" ]; then
        git clone --depth ${depth} "${mirrored_url}" "${target}"
    else
        git clone "${mirrored_url}" "${target}"
    fi
}

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}========================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}========================================${NC}\n"; }

# ==================== 初始化 conda ====================
init_conda() {
    log_info "初始化 conda..."
    eval "$(conda shell.bash hook)"
}

# ==================== 步骤 1: 创建 conda 环境 ====================
step1_create_env() {
    log_step "步骤 1/9: 创建 conda 环境 (Python ${PYTHON_VERSION})"

    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "conda 环境 '${CONDA_ENV_NAME}' 已存在，跳过创建"
    else
        log_info "创建 conda 环境: ${CONDA_ENV_NAME} (Python ${PYTHON_VERSION})"
        conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
        log_info "conda 环境创建完成"
    fi

    conda activate ${CONDA_ENV_NAME}
    log_info "当前 Python: $(python --version)"
    log_info "当前 Python 路径: $(which python)"
}

# ==================== 步骤 2: 安装 PyTorch ====================
step2_install_pytorch() {
    log_step "步骤 2/9: 安装 PyTorch 2.8.0 + CUDA 12.8"

    conda activate ${CONDA_ENV_NAME}

    if python -c "import torch; assert torch.__version__ == '2.8.0'" 2>/dev/null; then
        log_info "PyTorch 2.8.0 已安装，跳过"
    else
        log_info "安装 PyTorch 2.8.0 (CUDA 12.8)..."
        pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cu128
        log_info "PyTorch 安装完成"
    fi

    log_info "验证 PyTorch:"
    python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
}

# ==================== 步骤 3: 安装项目依赖 ====================
step3_install_requirements() {
    log_step "步骤 3/9: 安装项目 Python 依赖 + sox"

    conda activate ${CONDA_ENV_NAME}

    log_info "安装 requirements.txt..."
    cd ${PROJECT_DIR}
    pip install -r requirements.txt

    log_info "安装 sox (conda-forge)..."
    conda install conda-forge::sox -y

    log_info "项目依赖安装完成"
}

# ==================== 步骤 4: 安装 SageAttention ====================
step4_install_sageattention() {
    log_step "步骤 4/9: 安装 SageAttention v2.2.0 (FP8 attention kernel)"

    conda activate ${CONDA_ENV_NAME}

    if python -c "import sageattention" 2>/dev/null; then
        log_info "SageAttention 已安装，跳过"
    else
        mkdir -p ${TMP_DIR}
        cd ${TMP_DIR}

        if [ -d "SageAttention" ]; then
            log_info "SageAttention 目录已存在，更新..."
            cd SageAttention
            git fetch --all || log_warn "git fetch 失败, 继续使用本地版本"
        else
            log_info "克隆 SageAttention..."
            git_clone https://github.com/thu-ml/SageAttention.git SageAttention
            cd SageAttention
        fi

        git checkout v2.2.0
        log_info "编译安装 SageAttention..."
        python setup.py install

        cd ${PROJECT_DIR}
        log_info "SageAttention 安装完成"
    fi
}

# ==================== 步骤 5: 安装 vllm ====================
step5_install_vllm() {
    log_step "步骤 5/9: 安装 vllm 0.11.0 (FP8 GEMM kernel)"

    conda activate ${CONDA_ENV_NAME}

    if python -c "import vllm; print(vllm.__version__)" 2>/dev/null | grep -q "0.11.0"; then
        log_info "vllm 0.11.0 已安装，跳过"
    else
        log_info "安装 vllm 0.11.0..."
        pip install vllm==0.11.0
        log_info "vllm 安装完成"
    fi
}

# ==================== 步骤 6: 安装 LightX2V (VAE) ====================
step6_install_lightx2v() {
    log_step "步骤 6/9: 安装 LightX2V (VAE)"

    conda activate ${CONDA_ENV_NAME}

    if python -c "from lightx2v.models.video_encoders.hf.wan.vae import WanVAE" 2>/dev/null; then
        log_info "LightX2V VAE 已安装，跳过"
    else
        mkdir -p ${TMP_DIR}
        cd ${TMP_DIR}

        if [ -d "LightX2V" ]; then
            log_info "LightX2V 目录已存在，更新..."
            cd LightX2V
            git pull || log_warn "git pull 失败, 继续使用本地版本"
        else
            log_info "克隆 LightX2V..."
            git_clone https://github.com/ModelTC/LightX2V.git LightX2V
            cd LightX2V
        fi

        log_info "安装 LightX2V VAE..."
        python setup_vae.py install

        cd ${PROJECT_DIR}
        log_info "LightX2V VAE 安装完成"
    fi
}

# ==================== 步骤 7: 安装 lightx2v_kernel (FP4) ====================
step7_install_lightx2v_kernel() {
    log_step "步骤 7/9: 安装 lightx2v_kernel (NVFP4 GEMM, Blackwell FP4 支持)"

    conda activate ${CONDA_ENV_NAME}

    if python -c "from lightx2v_kernel.gemm import scaled_nvfp4_quant" 2>/dev/null; then
        log_info "lightx2v_kernel 已安装，跳过"
    else
        log_info "安装编译依赖..."
        pip install scikit_build_core uv

        mkdir -p ${TMP_DIR}
        cd ${TMP_DIR}

        # 下载 CUTLASS (NVFP4 算子依赖)
        if [ ! -d "cutlass" ]; then
            log_info "克隆 CUTLASS (浅克隆)..."
            git_clone https://github.com/NVIDIA/cutlass.git cutlass 1
        fi
        CUTLASS_PATH="${TMP_DIR}/cutlass"

        # 确保 LightX2V 已克隆
        if [ ! -d "LightX2V" ]; then
            log_info "克隆 LightX2V..."
            git_clone https://github.com/ModelTC/LightX2V.git LightX2V
        fi

        cd LightX2V/lightx2v_kernel
        log_info "编译 lightx2v_kernel (可能需要几分钟)..."
        MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
            uv build --wheel \
            -Cbuild-dir=build . \
            -Ccmake.define.CUTLASS_PATH=${CUTLASS_PATH} \
            --verbose \
            --color=always \
            --no-build-isolation

        log_info "安装 lightx2v_kernel whl..."
        pip install dist/*whl --force-reinstall --no-deps

        cd ${PROJECT_DIR}
        log_info "lightx2v_kernel 安装完成"
    fi
}

# ==================== 步骤 8: 下载模型权重 ====================
step8_download_models() {
    log_step "步骤 8/9: 下载模型权重"

    conda activate ${CONDA_ENV_NAME}
    mkdir -p ${MODEL_DIR}

    # 安装下载工具
    pip install modelscope huggingface_hub

    # ---------- 8.1 下载 SoulX-LiveAct 主模型 ----------
    LIVEACT_DIR="${MODEL_DIR}/LiveAct"
    if [ -d "${LIVEACT_DIR}" ] && [ "$(ls -A ${LIVEACT_DIR} 2>/dev/null)" ]; then
        log_info "SoulX-LiveAct 模型已存在，跳过下载: ${LIVEACT_DIR}"
    else
        log_info "下载 SoulX-LiveAct 主模型 (从 ModelScope)..."
        mkdir -p ${LIVEACT_DIR}
        modelscope download --model Soul-AILab/LiveAct --local_dir ${LIVEACT_DIR}
        log_info "SoulX-LiveAct 模型下载完成: ${LIVEACT_DIR}"
    fi

    # ---------- 8.2 下载 chinese-wav2vec2-base ----------
    WAV2VEC_DIR="${MODEL_DIR}/chinese-wav2vec2-base"
    if [ -d "${WAV2VEC_DIR}" ] && [ "$(ls -A ${WAV2VEC_DIR} 2>/dev/null)" ]; then
        log_info "chinese-wav2vec2-base 已存在，跳过下载: ${WAV2VEC_DIR}"
    else
        log_info "下载 chinese-wav2vec2-base..."
        mkdir -p ${WAV2VEC_DIR}

        # 优先尝试 ModelScope
        if modelscope download --model TencentGameMate/chinese-wav2vec2-base --local_dir ${WAV2VEC_DIR} 2>/dev/null; then
            log_info "chinese-wav2vec2-base 下载完成 (ModelScope): ${WAV2VEC_DIR}"
        else
            log_warn "ModelScope 下载失败，尝试 HuggingFace 镜像..."
            export HF_ENDPOINT=https://hf-mirror.com
            huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ${WAV2VEC_DIR}
            log_info "chinese-wav2vec2-base 下载完成 (HF镜像): ${WAV2VEC_DIR}"
        fi
    fi

    log_info "模型目录结构:"
    ls -la ${MODEL_DIR}
}

# ==================== 步骤 9: 验证安装 ====================
step9_verify() {
    log_step "步骤 9/9: 验证安装"

    conda activate ${CONDA_ENV_NAME}

    log_info "========== 环境验证 =========="

    # Python
    python --version

    # PyTorch + CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'GPU compute capability: {cap[0]}.{cap[1]} (Blackwell >= 10.0)')
"

    # SageAttention
    python -c "import sageattention; print('SageAttention: OK')" 2>/dev/null || log_warn "SageAttention 未安装"

    # vllm
    python -c "import vllm; print(f'vllm: {vllm.__version__}')" 2>/dev/null || log_warn "vllm 未安装"

    # LightX2V VAE
    python -c "from lightx2v.models.video_encoders.hf.wan.vae import WanVAE; print('LightX2V VAE: OK')" 2>/dev/null || log_warn "LightX2V VAE 未安装"

    # lightx2v_kernel (FP4)
    python -c "from lightx2v_kernel.gemm import scaled_nvfp4_quant; print('lightx2v_kernel (FP4): OK')" 2>/dev/null || log_warn "lightx2v_kernel 未安装 (FP4 不可用)"

    # 模型权重
    if [ -d "${MODEL_DIR}/LiveAct" ]; then
        log_info "LiveAct 模型目录: ${MODEL_DIR}/LiveAct"
        ls "${MODEL_DIR}/LiveAct" | head -5
    else
        log_warn "LiveAct 模型未下载"
    fi

    if [ -d "${MODEL_DIR}/chinese-wav2vec2-base" ]; then
        log_info "wav2vec2 模型目录: ${MODEL_DIR}/chinese-wav2vec2-base"
        ls "${MODEL_DIR}/chinese-wav2vec2-base" | head -5
    else
        log_warn "chinese-wav2vec2-base 未下载"
    fi

    echo ""
    log_info "========== 验证完成 =========="
    echo ""
    log_info "启动命令 (FP4 模式, RTX PRO 6000):"
    echo ""
    echo "  conda activate liveact"
    echo "  cd ~/SoulX-LiveAct"
    echo "  USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 \\"
    echo "  python demo.py \\"
    echo "      --ckpt_dir ./models/LiveAct \\"
    echo "      --wav2vec_dir ./models/chinese-wav2vec2-base \\"
    echo "      --size 416*720 \\"
    echo "      --fps 20 \\"
    echo "      --fp4_gemm \\"
    echo "      --port 5001"
    echo ""
}

# ==================== 主入口 ====================
main() {
    local cmd="${1:-all}"

    init_conda

    case "$cmd" in
        env)
            step1_create_env
            step2_install_pytorch
            step3_install_requirements
            step4_install_sageattention
            step5_install_vllm
            ;;
        kernel)
            step6_install_lightx2v
            step7_install_lightx2v_kernel
            ;;
        models)
            step8_download_models
            ;;
        verify)
            step9_verify
            ;;
        all)
            step1_create_env
            step2_install_pytorch
            step3_install_requirements
            step4_install_sageattention
            step5_install_vllm
            step6_install_lightx2v
            step7_install_lightx2v_kernel
            step8_download_models
            step9_verify
            ;;
        *)
            echo "用法: bash setup.sh [env|kernel|models|verify|all]"
            echo ""
            echo "  env     - 步骤 1-5: 创建 conda 环境 + 安装 Python 依赖 (PyTorch/requirements/SageAttention/vllm)"
            echo "  kernel  - 步骤 6-7: 安装 LightX2V VAE + lightx2v_kernel (FP4)"
            echo "  models  - 步骤 8:   下载模型权重 (LiveAct + chinese-wav2vec2-base)"
            echo "  verify  - 步骤 9:   验证安装"
            echo "  all     - 全部执行 (默认)"
            exit 1
            ;;
    esac
}

main "$@"
