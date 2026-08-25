#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pyenv_bin="/home/dwb/.pyenv/bin/pyenv"
export PYENV_VERSION="${PYENV_VERSION:-miniconda3-3.12-25.11.1-1}"
sim_python="${project_root}/.venv-octo-simpler/bin/python"
checkpoint="/data/dwb/models/octo-small-pytorch-official"
device="cuda:0"
precision="bf16"
sim_device="cuda:0"
output_dir="${project_root}/outputs/octo_small_official_pytorch_simpler_eval"
server_timeout="600"
forwarded=()
overwrite=0
preflight_only=0

declare -A seen=()

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_simpler_octo_small_official_pytorch.sh [launcher options] [evaluation options]

Launcher options:
  --pyenv-bin PATH       model Python via pyenv (default: /home/dwb/.pyenv/bin/pyenv)
  --sim-python PATH      SimplerEnv Python (default: .venv-octo-simpler/bin/python)
  --checkpoint PATH      standalone official checkpoint
                         (default: /data/dwb/models/octo-small-pytorch-official)
  --device DEVICE        single model GPU (default: cuda:0)
  --precision VALUE      bf16 or fp32 (default: bf16)
  --sim-device DEVICE    simulator renderer GPU (default: cuda:0)
  --output-dir PATH      output directory
                         (default: outputs/octo_small_official_pytorch_simpler_eval)
  --server-timeout SEC   maximum model startup wait (default: 600)
  -h, --help             show this help

Evaluation options:
  --tasks all|spoon,carrot,stack,eggplant
  --action-postprocessing octo_temporal_ensemble_v1|first_action
  --preflight-only --smoke-test --overwrite
  --save-videos-path PATH --video-fps FPS

The checkpoint is self-contained, and this launcher never converts implicitly.
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

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --base-model|--base-model=*|--statistics|--statistics=*|--model-devices|--model-devices=*)
      fail "$1 is not supported by the official PyTorch baseline"
      ;;
    --pyenv-bin|--sim-python|--checkpoint|--device|--precision|--sim-device|--output-dir|--server-timeout)
      option="$1"
      set_once "${option}"
      (($# >= 2)) || fail "${option} requires a non-empty value"
      [[ -n "$2" ]] || fail "${option} requires a non-empty value"
      case "${option}" in
        --pyenv-bin) pyenv_bin="$2" ;;
        --sim-python) sim_python="$2" ;;
        --checkpoint) checkpoint="$2" ;;
        --device) device="$2" ;;
        --precision) precision="$2" ;;
        --sim-device) sim_device="$2" ;;
        --output-dir) output_dir="$2" ;;
        --server-timeout) server_timeout="$2" ;;
      esac
      shift 2
      ;;
    --pyenv-bin=*|--sim-python=*|--checkpoint=*|--device=*|--precision=*|--sim-device=*|--output-dir=*|--server-timeout=*)
      option="${1%%=*}"
      value="${1#*=}"
      set_once "${option}"
      [[ -n "${value}" ]] || fail "${option} requires a non-empty value"
      case "${option}" in
        --pyenv-bin) pyenv_bin="${value}" ;;
        --sim-python) sim_python="${value}" ;;
        --checkpoint) checkpoint="${value}" ;;
        --device) device="${value}" ;;
        --precision) precision="${value}" ;;
        --sim-device) sim_device="${value}" ;;
        --output-dir) output_dir="${value}" ;;
        --server-timeout) server_timeout="${value}" ;;
      esac
      shift
      ;;
    --socket|--socket=*|--auth-key-hex|--auth-key-hex=*)
      fail "$1 is managed by this launcher"
      ;;
    --overwrite)
      overwrite=1
      forwarded+=("$1")
      shift
      ;;
    --preflight-only)
      preflight_only=1
      forwarded+=("$1")
      shift
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

[[ -d "${checkpoint}" ]] || fail "official checkpoint is missing: ${checkpoint}; run scripts/convert_octo_small_official_to_pytorch.sh"
[[ "${device}" =~ ^cuda:[0-9]+$ ]] || fail "--device must name one GPU as cuda:<index>"
[[ "${sim_device}" =~ ^cuda:[0-9]+$ ]] || fail "--sim-device must match cuda:<index>"
[[ "${precision}" == "bf16" || "${precision}" == "fp32" ]] || fail "--precision must be bf16 or fp32"
[[ "${server_timeout}" =~ ^[1-9][0-9]*$ ]] || fail "--server-timeout must be a positive integer"
[[ -x "${pyenv_bin}" ]] || fail "pyenv executable is not executable: ${pyenv_bin}"
[[ -x "${sim_python}" ]] || fail "simulator Python is not executable: ${sim_python}"

if ((overwrite == 0)); then
  protected_outputs=("${output_dir}/failure.json" "${output_dir}/model-server.log")
  if ((preflight_only)); then
    protected_outputs+=("${output_dir}/preflight.json")
  else
    protected_outputs+=(
      "${output_dir}/results.json"
      "${output_dir}/episodes.jsonl"
      "${output_dir}/episodes.partial.jsonl"
    )
  fi
  for protected_output in "${protected_outputs[@]}"; do
    [[ ! -e "${protected_output}" ]] || fail "evaluation output already exists; use --overwrite: ${protected_output}"
  done
fi

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
if ! "${pyenv_bin}" exec python -m octo_small_official_pytorch.checkpoint_cli "${checkpoint}" >/dev/null; then
  fail "checkpoint is neither a valid official base artifact nor an official fine-tune checkpoint"
fi

mkdir -p "${output_dir}"
model_log="${output_dir}/model-server.log"
ipc_dir="$(mktemp -d "${TMPDIR:-/tmp}/octo-official-simpler.XXXXXXXX")"
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
  export PYTHONPATH="${project_root}/src"
  exec "${pyenv_bin}" exec python -m octo_small_official_pytorch.server \
    --socket "${socket_path}" \
    --auth-key-hex "${auth_key_hex}" \
    --checkpoint "${checkpoint}" \
    --device "${device}" \
    --precision "${precision}"
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
    printf 'error: model server exited before ready (status %s); log: %s\n' "${model_status}" "${model_log}" >&2
    tail -n 40 "${model_log}" >&2 || true
    ((model_status != 0)) || model_status=1
    exit "${model_status}"
  fi
  sleep 0.1
done

if [[ ! -S "${socket_path}" ]]; then
  printf 'error: timed out after %ss waiting for model server; log: %s\n' "${server_timeout}" "${model_log}" >&2
  exit 1
fi

simpler_root="${project_root}/third_party/SimplerEnv"
maniskill_root="${simpler_root}/ManiSkill2_real2sim"
sim_status=0
(
  export PYTHONPATH="${project_root}/src:${simpler_root}:${maniskill_root}"
  export MS2_REAL2SIM_ASSET_DIR="${maniskill_root}/data"
  exec "${sim_python}" -m octo_small_official_pytorch.evaluate_simpler \
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
  model_pid=""
  if ((first_status != 0)); then
    terminate_child "${sim_pid}"
    sim_pid=""
    printf 'error: model server failed during evaluation (status %s); log: %s\n' "${first_status}" "${model_log}" >&2
    tail -n 40 "${model_log}" >&2 || true
    exit "${first_status}"
  fi
  wait "${sim_pid}" || sim_status=$?
  sim_pid=""
else
  sim_status="${first_status}"
  sim_pid=""
fi

exit "${sim_status}"
