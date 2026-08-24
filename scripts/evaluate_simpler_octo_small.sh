#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pyenv_bin="/home/dwb/.pyenv/bin/pyenv"
sim_python="${project_root}/.venv-octo-simpler/bin/python"
checkpoint=""
base_model=""
statistics=""
device="cuda:0"
device_explicit=0
model_devices=""
precision="bf16"
sim_device="cuda:0"
output_dir="${project_root}/outputs/octo_small_bridge_simpler_eval"
server_timeout=600
forwarded=()

declare -A seen=()

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_simpler_octo_small.sh [launcher options] [evaluation options]

Launcher options:
  --pyenv-bin PATH       pyenv executable (default: /home/dwb/.pyenv/bin/pyenv)
  --sim-python PATH      SimplerEnv Python (default: .venv-octo-simpler/bin/python)
  --checkpoint PATH      Octo Bridge step-XXXXXXXX checkpoint directory (required)
  --base-model PATH      converted Octo-small PyTorch base model (required)
  --statistics PATH      optional normalization.json override
  --device DEVICE        model service CUDA device (default: cuda:0)
  --model-devices LIST   parallel model replicas, e.g. cuda:0,cuda:1
                         cannot be combined with --device
  --precision VALUE      model precision: bf16 or fp32 (default: bf16)
  --sim-device DEVICE    simulator renderer CUDA device (default: cuda:0)
  --output-dir PATH      evaluation output and persistent model-server.log
  --server-timeout SEC   maximum model load wait (default: 600)
  -h, --help             show this help without starting either process

Evaluation options are forwarded to octo_small_bridge.evaluate_simpler, including:
  --tasks TASKS --action-horizon 1
  --action-postprocessing octo_temporal_ensemble_v1|first_action (default: octo_temporal_ensemble_v1)
  --preflight-only --smoke-test
  --save-videos-path PATH --video-fps FPS --overwrite

The removed --python option is not accepted; use --sim-python instead.
EOF
}

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 2
}

set_once() {
  local option="$1"
  [[ -z "${seen[${option}]:-}" ]] || fail "${option} may only be specified once"
  seen["${option}"]=1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || fail "${option} requires a non-empty value"
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --python|--python=*)
      fail "--python has been removed; use --sim-python"
      ;;
    --pyenv-bin|--sim-python|--checkpoint|--base-model|--statistics|--device|--model-devices|--precision|--sim-device|--output-dir|--server-timeout)
      option="$1"
      set_once "${option}"
      (($# >= 2)) || fail "${option} requires a non-empty value"
      require_value "${option}" "$2"
      case "${option}" in
        --pyenv-bin) pyenv_bin="$2" ;;
        --sim-python) sim_python="$2" ;;
        --checkpoint) checkpoint="$2" ;;
        --base-model) base_model="$2" ;;
        --statistics) statistics="$2" ;;
        --device) device="$2"; device_explicit=1 ;;
        --model-devices) model_devices="$2" ;;
        --precision) precision="$2" ;;
        --sim-device) sim_device="$2" ;;
        --output-dir) output_dir="$2" ;;
        --server-timeout) server_timeout="$2" ;;
      esac
      shift 2
      ;;
    --pyenv-bin=*|--sim-python=*|--checkpoint=*|--base-model=*|--statistics=*|--device=*|--model-devices=*|--precision=*|--sim-device=*|--output-dir=*|--server-timeout=*)
      option="${1%%=*}"
      value="${1#*=}"
      set_once "${option}"
      require_value "${option}" "${value}"
      case "${option}" in
        --pyenv-bin) pyenv_bin="${value}" ;;
        --sim-python) sim_python="${value}" ;;
        --checkpoint) checkpoint="${value}" ;;
        --base-model) base_model="${value}" ;;
        --statistics) statistics="${value}" ;;
        --device) device="${value}"; device_explicit=1 ;;
        --model-devices) model_devices="${value}" ;;
        --precision) precision="${value}" ;;
        --sim-device) sim_device="${value}" ;;
        --output-dir) output_dir="${value}" ;;
        --server-timeout) server_timeout="${value}" ;;
      esac
      shift
      ;;
    --action-horizon)
      set_once "--action-horizon"
      (($# >= 2)) || fail "--action-horizon requires a non-empty value"
      [[ "$2" == "1" ]] || fail "--action-horizon must be 1 for stepwise inference"
      forwarded+=("$1" "$2")
      shift 2
      ;;
    --action-horizon=*)
      set_once "--action-horizon"
      value="${1#*=}"
      [[ "${value}" == "1" ]] || fail "--action-horizon must be 1 for stepwise inference"
      forwarded+=("$1")
      shift
      ;;
    --socket|--socket=*|--auth-key-hex|--auth-key-hex=*|--shard-index|--shard-index=*|--shard-count|--shard-count=*|--rng-scope|--rng-scope=*)
      fail "$1 is managed by this launcher"
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

[[ -n "${checkpoint}" ]] || fail "--checkpoint is required"
[[ -n "${base_model}" ]] || fail "--base-model is required"
[[ "${precision}" == "bf16" || "${precision}" == "fp32" ]] ||
  fail "--precision must be bf16 or fp32"
[[ "${server_timeout}" =~ ^[1-9][0-9]*$ ]] ||
  fail "--server-timeout must be a positive integer"
[[ "${sim_device}" =~ ^cuda:[0-9]+$ ]] ||
  fail "--sim-device must match cuda:<non-negative decimal integer>"
if [[ -n "${model_devices}" ]]; then
  ((device_explicit == 0)) || fail "--model-devices cannot be combined with --device"
  [[ "${model_devices}" =~ ^cuda:[0-9]+(,cuda:[0-9]+)*$ ]] ||
    fail "--model-devices must be a comma-separated list of unique cuda:<index> values"
  declare -A seen_model_devices=()
  IFS=',' read -r -a parsed_model_devices <<<"${model_devices}"
  for model_device in "${parsed_model_devices[@]}"; do
    [[ -z "${seen_model_devices[${model_device}]:-}" ]] ||
      fail "--model-devices must be a comma-separated list of unique cuda:<index> values"
    seen_model_devices["${model_device}"]=1
  done
fi
[[ -x "${pyenv_bin}" ]] || fail "pyenv executable is not executable: ${pyenv_bin}"
[[ -x "${sim_python}" ]] || fail "simulator Python is not executable: ${sim_python}"

if [[ -n "${model_devices}" ]]; then
  parallel_command=(
    "${pyenv_bin}" exec python -m octo_small_bridge.parallel_evaluation
    --sim-python "${sim_python}"
    --checkpoint "${checkpoint}"
    --base-model "${base_model}"
    --model-devices "${model_devices}"
    --sim-device "${sim_device}"
    --precision "${precision}"
    --output-dir "${output_dir}"
    --server-timeout "${server_timeout}"
  )
  if [[ -n "${statistics}" ]]; then
    parallel_command+=(--statistics "${statistics}")
  fi
  parallel_command+=(-- "${forwarded[@]}")
  simpler_root="${project_root}/third_party/SimplerEnv"
  maniskill_root="${simpler_root}/ManiSkill2_real2sim"
  export PYTHONPATH="${project_root}/src:${simpler_root}:${maniskill_root}"
  export MS2_REAL2SIM_ASSET_DIR="${maniskill_root}/data"
  exec "${parallel_command[@]}"
fi

mkdir -p "${output_dir}"
model_log="${output_dir}/model-server.log"
ipc_dir="$(mktemp -d "${TMPDIR:-/tmp}/octo-simpler.XXXXXXXX")"
chmod 700 "${ipc_dir}"
socket_path="${ipc_dir}/model.sock"
auth_key_hex="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
model_pid=""
sim_pid=""
cleanup_started=0

terminate_child() {
  local pid="$1"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  for _attempt in {1..100}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  if ((cleanup_started)); then
    return 0
  fi
  cleanup_started=1
  trap - EXIT INT TERM
  terminate_child "${sim_pid}"
  terminate_child "${model_pid}"
  rm -f "${socket_path}"
  rmdir "${ipc_dir}" 2>/dev/null || true
  return "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

model_command=(
  "${pyenv_bin}" exec python -m octo_small_bridge.server
  --socket "${socket_path}"
  --auth-key-hex "${auth_key_hex}"
  --checkpoint "${checkpoint}"
  --base-model "${base_model}"
  --device "${device}"
  --precision "${precision}"
)
if [[ -n "${statistics}" ]]; then
  model_command+=(--statistics "${statistics}")
fi

(
  export PYTHONPATH="${project_root}/src"
  exec "${model_command[@]}"
) >"${model_log}" 2>&1 &
model_pid=$!

max_attempts=$((server_timeout * 10))
for ((attempt = 0; attempt < max_attempts; attempt++)); do
  if [[ -S "${socket_path}" ]]; then
    break
  fi
  if ! kill -0 "${model_pid}" 2>/dev/null; then
    model_status=0
    wait "${model_pid}" || model_status=$?
    model_pid=""
    printf 'error: model server exited before becoming ready (status %s); log: %s\n' \
      "${model_status}" "${model_log}" >&2
    tail -n 40 "${model_log}" >&2 || true
    ((model_status != 0)) || model_status=1
    exit "${model_status}"
  fi
  sleep 0.1
done

if [[ ! -S "${socket_path}" ]]; then
  printf 'error: timed out after %ss waiting for model server; log: %s\n' \
    "${server_timeout}" "${model_log}" >&2
  exit 1
fi

simpler_root="${project_root}/third_party/SimplerEnv"
maniskill_root="${simpler_root}/ManiSkill2_real2sim"
sim_status=0
(
  export PYTHONPATH="${project_root}/src:${simpler_root}:${maniskill_root}"
  export MS2_REAL2SIM_ASSET_DIR="${maniskill_root}/data"
  exec "${sim_python}" -m octo_small_bridge.evaluate_simpler \
    --socket "${socket_path}" \
    --auth-key-hex "${auth_key_hex}" \
    --output-dir "${output_dir}" \
    --sim-device "${sim_device}" \
    "${forwarded[@]}"
) &
sim_pid=$!
completed_pid=""
first_status=0
wait -n -p completed_pid "${sim_pid}" "${model_pid}" || first_status=$?

if [[ "${completed_pid}" == "${model_pid}" ]]; then
  model_status="${first_status}"
  model_pid=""
  if ((model_status != 0)); then
    terminate_child "${sim_pid}"
    sim_pid=""
    printf 'error: model server failed during evaluation (status %s); log: %s\n' \
      "${model_status}" "${model_log}" >&2
    tail -n 40 "${model_log}" >&2 || true
    exit "${model_status}"
  fi
  wait "${sim_pid}" || sim_status=$?
  sim_pid=""
else
  sim_status="${first_status}"
  sim_pid=""
fi

if [[ -n "${model_pid}" ]] && ! kill -0 "${model_pid}" 2>/dev/null; then
  model_status=0
  wait "${model_pid}" || model_status=$?
  model_pid=""
  if ((sim_status == 0 && model_status != 0)); then
    printf 'error: model server failed during evaluation (status %s); log: %s\n' \
      "${model_status}" "${model_log}" >&2
    sim_status="${model_status}"
  fi
fi

exit "${sim_status}"
