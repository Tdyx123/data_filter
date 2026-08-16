#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pyenv_bin="/home/dwb/.pyenv/bin/pyenv"
pyenv_version="miniconda3-3.12-25.11.1-1"
sim_python="${project_root}/.venv-octo-simpler/bin/python"
model_dir="/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1"
base_model="/data/dwb/models/Qwen3-VL-4B-Instruct"
device="cuda:0"
output_dir="${project_root}/outputs/starvla_simpler_eval"
server_timeout=600
forwarded=()

declare -A seen=()

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_simpler_starvla.sh [launcher options] [evaluation options]

Launcher options:
  --pyenv-bin PATH       pyenv executable (default: /home/dwb/.pyenv/bin/pyenv)
  --pyenv-version NAME   model runtime (default: miniconda3-3.12-25.11.1-1)
  --sim-python PATH      SimplerEnv Python (default: .venv-octo-simpler/bin/python)
  --model-dir PATH       StarVLA checkpoint directory
  --base-model PATH      local Qwen3-VL config/processor directory
  --device DEVICE        model device (default: cuda:0)
  --output-dir PATH      evaluation output and persistent model-server.log
  --server-timeout SEC   maximum model load wait (default: 600)
  -h, --help             show this help without starting either process

Evaluation options are forwarded to starvla_bridge.evaluate_simpler, including:
  --tasks TASKS --action-horizon 1 --preflight-only --smoke-test
  --save-videos-path PATH --overwrite
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
    --pyenv-bin|--pyenv-version|--sim-python|--model-dir|--base-model|--device|--output-dir|--server-timeout)
      option="$1"
      set_once "${option}"
      (($# >= 2)) || fail "${option} requires a non-empty value"
      require_value "${option}" "$2"
      case "${option}" in
        --pyenv-bin) pyenv_bin="$2" ;;
        --pyenv-version) pyenv_version="$2" ;;
        --sim-python) sim_python="$2" ;;
        --model-dir) model_dir="$2" ;;
        --base-model) base_model="$2" ;;
        --device) device="$2" ;;
        --output-dir) output_dir="$2" ;;
        --server-timeout) server_timeout="$2" ;;
      esac
      shift 2
      ;;
    --pyenv-bin=*|--pyenv-version=*|--sim-python=*|--model-dir=*|--base-model=*|--device=*|--output-dir=*|--server-timeout=*)
      option="${1%%=*}"
      value="${1#*=}"
      set_once "${option}"
      require_value "${option}" "${value}"
      case "${option}" in
        --pyenv-bin) pyenv_bin="${value}" ;;
        --pyenv-version) pyenv_version="${value}" ;;
        --sim-python) sim_python="${value}" ;;
        --model-dir) model_dir="${value}" ;;
        --base-model) base_model="${value}" ;;
        --device) device="${value}" ;;
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
    --socket|--socket=*|--auth-key-hex|--auth-key-hex=*)
      fail "$1 is managed by this launcher"
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

[[ "${server_timeout}" =~ ^[1-9][0-9]*$ ]] || fail "--server-timeout must be a positive integer"
[[ -x "${pyenv_bin}" ]] || fail "pyenv executable is not executable: ${pyenv_bin}"
[[ -x "${sim_python}" ]] || fail "simulator Python is not executable: ${sim_python}"

mkdir -p "${output_dir}"
model_log="${output_dir}/model-server.log"
ipc_dir="$(mktemp -d "${TMPDIR:-/tmp}/starvla-simpler.XXXXXXXX")"
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

(
  export PYENV_VERSION="${pyenv_version}"
  export PYTHONPATH="${project_root}/src"
  exec "${pyenv_bin}" exec python -m starvla_bridge.server \
    --socket "${socket_path}" \
    --auth-key-hex "${auth_key_hex}" \
    --model-dir "${model_dir}" \
    --base-model "${base_model}" \
    --device "${device}"
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
  exec "${sim_python}" -m starvla_bridge.evaluate_simpler \
    --socket "${socket_path}" \
    --auth-key-hex "${auth_key_hex}" \
    --output-dir "${output_dir}" \
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
