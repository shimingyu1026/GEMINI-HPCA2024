#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# Gemini EyerissCore / TPU-like mapping runner
#
# 用法:
#   ./run_eyeriss_tpu.sh
#
# 也可以用环境变量覆盖默认参数:
#   NETWORK_ID=2 MAC_PER_CORE=2048 UL2_KB=4096 ./run_eyeriss_tpu.sh
#
# 说明:
# - 这个脚本固定使用 `mm=1`，也就是 EyerissCore 路径。
# - 这个脚本只跑一次固定架构配置，不做 DSE 架构探索。
# - 当前代码里 `mm=1` 会自动强制成 “每个 chiplet 只有一个 core”。
#   所以这里直接令:
#     XCUT = XX
#     YCUT = YY
# - 当前代码里 `_mac_dim` 表示 “每个 core 的总 PE/MAC 数”，不是边长。
#   程序会自动把它分解成二维阵列:
#     1024 -> 32x32
#     2048 -> 64x32
# -----------------------------------------------------------------------------

# 是否在运行前重新编译。
# 0: 仅当 `build/stschedule` 不存在时编译
# 1: 每次都重新编译
REBUILD="${REBUILD:-0}"

# 工艺节点。
# 可选: 7 / 12
TECH="${TECH:-12}"

# 网络编号。
# 对应关系:
#   0=darknet19
#   1=vgg
#   2=resnet50
#   3=googlenet
#   4=resnet101
#   5=densenet
#   6=incep_resnet
#   7=gnmt
#   8=lstm
#   9=zfnet
#   10=transformer
#   11=transformer_cell
#   12=pnasnet
#   13=resnext50
#   14=resnet152
#   15=BERT_block
#   16=GPT2_prefill_block
#   17=GPT2_decode_block
#   18=resnet_small
NETWORK_ID="${NETWORK_ID:-18}"

# 核心网格尺寸。
# 在 `mm=1` 下，这同时也是 chiplet 网格尺寸，因为 1 chiplet = 1 core。
XX="${XX:-1}"
YY="${YY:-1}"

# Cluster stride。
# 一般保持 1。
STRIDE="${STRIDE:-1}"

# batch size。
BATCH_SIZE="${BATCH_SIZE:-4}"

# 映射搜索轮数。
# 这里只做单次网络映射，不做架构探索。
ROUNDS="${ROUNDS:-10}"

# 优化目标。
#   1  = energy * time
#   0  = time only
#  -1  = energy only
OBJECTIVE="${OBJECTIVE:-0}"

# 封装类型。
# 可选: OS / FO / SI
PACKAGE_TYPE="${PACKAGE_TYPE:-OS}"

# Die-to-die IO 类型。
# 可选: UCIe / XSR / USR
IO_TYPE="${IO_TYPE:-UCIe}"

# NoP 带宽。
# 单位与仓库原始脚本保持一致。
NOP_BW="${NOP_BW:-16}"

# DDR 类型。
# 常用: GDDR6X
DDR_TYPE="${DDR_TYPE:-GDDR6X}"

# DRAM 带宽。
# 保持与原始 DSE 脚本同一输入单位。
# 例如 8192 表示脚本里常见的一档配置。
DRAM_BW="${DRAM_BW:-8192}"

# NoC 带宽。
NOC_BW="${NOC_BW:-16}"

# 每个 core 的总 PE/MAC 数。
# 当前 `mm=1` 下程序会自动分解成二维阵列。
# 例如:
#   1024 -> 32x32
#   2048 -> 64x32
MAC_PER_CORE="${MAC_PER_CORE:-16384}"

# 每个 core 的 UL2 容量，单位 KB。
# 当前 `mm=1` 下，这个值会映射到 EyerissCore 的 `ul2.Size`。
UL2_KB="${UL2_KB:-10240}"

# -----------------------------------------------------------------------------
# 一般不需要手动改下面这些派生参数
# -----------------------------------------------------------------------------

MM=1
XCUT="$XX"
YCUT="$YY"

network_name_from_id() {
  case "$1" in
    0) echo "darknet19" ;;
    1) echo "vgg" ;;
    2) echo "resnet50" ;;
    3) echo "googlenet" ;;
    4) echo "resnet101" ;;
    5) echo "densenet" ;;
    6) echo "incep_resnet" ;;
    7) echo "gnmt" ;;
    8) echo "lstm" ;;
    9) echo "zfnet" ;;
    10) echo "transformer" ;;
    11) echo "transformer_cell" ;;
    12) echo "pnasnet" ;;
    13) echo "resnext50" ;;
    14) echo "resnet152" ;;
    15) echo "BERT_block" ;;
    16) echo "GPT2_prefill_block" ;;
    17) echo "GPT2_decode_block" ;;
    18) echo "resnet_small" ;;
    *) echo "network${1}" ;;
  esac
}

NETWORK_NAME="$(network_name_from_id "$NETWORK_ID")"

# 默认输出目录按 网络名 + 核心网格 + batch 数量 命名。
# `stschedule` 会把导出文件写到 `${RUN_DIR}/exports/`。
DEFAULT_RUN_DIR="$ROOT_DIR/tmp/${NETWORK_NAME}_${XX}x${YY}_b${BATCH_SIZE}"
RUN_DIR="${RUN_DIR:-$DEFAULT_RUN_DIR}"

# `total_tops` 这个输入字段沿用仓库原来的风格，单位相当于 “TOPS * 1024”。
# 对于 `mm=1`，总系统算力按:
#   total_tops = MAC_PER_CORE * XX * YY * 2
TOTAL_TOPS=$(( MAC_PER_CORE * XX * YY * 2 ))

RESULT_LOG="$RUN_DIR/result.log"
STDOUT_LOG="$RUN_DIR/stdout.log"
STDERR_LOG="$RUN_DIR/stderr.log"
PARAM_LOG="$RUN_DIR/params.txt"

mkdir -p "$RUN_DIR"

if [[ "$REBUILD" == "1" || ! -x "$ROOT_DIR/build/stschedule" ]]; then
  echo "[build] make -j4"
  make -j4
fi

INPUT_LINE="$TECH $MM $NETWORK_ID $XX $YY $STRIDE $BATCH_SIZE $ROUNDS $OBJECTIVE $XCUT $YCUT $PACKAGE_TYPE $IO_TYPE $NOP_BW $DDR_TYPE $DRAM_BW $NOC_BW $MAC_PER_CORE $UL2_KB $TOTAL_TOPS"

cat > "$PARAM_LOG" <<EOF
TECH=$TECH
MM=$MM
NETWORK_ID=$NETWORK_ID
NETWORK_NAME=$NETWORK_NAME
XX=$XX
YY=$YY
STRIDE=$STRIDE
BATCH_SIZE=$BATCH_SIZE
ROUNDS=$ROUNDS
OBJECTIVE=$OBJECTIVE
XCUT=$XCUT
YCUT=$YCUT
PACKAGE_TYPE=$PACKAGE_TYPE
IO_TYPE=$IO_TYPE
NOP_BW=$NOP_BW
DDR_TYPE=$DDR_TYPE
DRAM_BW=$DRAM_BW
NOC_BW=$NOC_BW
MAC_PER_CORE=$MAC_PER_CORE
UL2_KB=$UL2_KB
TOTAL_TOPS=$TOTAL_TOPS
INPUT_LINE=$INPUT_LINE
EOF

echo "[run] output dir: $RUN_DIR"
echo "[run] input: $INPUT_LINE"

printf '%s\n' "$INPUT_LINE" \
  | "$ROOT_DIR/build/stschedule" "$RESULT_LOG" \
  > "$STDOUT_LOG" \
  2> "$STDERR_LOG"

cat "$STDOUT_LOG"
if [[ -s "$STDERR_LOG" ]]; then
  cat "$STDERR_LOG" >&2
fi

echo
echo "[done] result log: $RESULT_LOG"
echo "[done] stdout log: $STDOUT_LOG"
echo "[done] stderr log: $STDERR_LOG"
echo "[done] params log: $PARAM_LOG"
echo "[done] exports dir: $RUN_DIR/exports"
