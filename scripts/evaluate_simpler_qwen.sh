#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="python3"
python_seen=0
checkpoint=""
checkpoint_seen=0
sim_device="cuda:0"
sim_device_seen=0
help_requested=0
forwarded=()

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 2
}

validate_sim_device() {
  [[ "$1" =~ ^cuda:[0-9]+$ ]] ||
    fail "--sim-device must match cuda:<non-negative decimal integer>"
}

while (($#)); do
  case "$1" in
    --python)
      ((python_seen == 0)) || fail "--python may only be specified once"
      (($# >= 2)) || fail "--python requires a non-empty value"
      [[ -n "$2" ]] || fail "--python requires a non-empty value"
      python_executable="$2"
      python_seen=1
      shift 2
      ;;
    --python=*)
      ((python_seen == 0)) || fail "--python may only be specified once"
      python_executable="${1#*=}"
      [[ -n "${python_executable}" ]] || fail "--python requires a non-empty value"
      python_seen=1
      shift
      ;;
    --checkpoint)
      ((checkpoint_seen == 0)) || fail "--checkpoint may only be specified once"
      (($# >= 2)) || fail "--checkpoint requires a non-empty value"
      [[ -n "$2" ]] || fail "--checkpoint requires a non-empty value"
      checkpoint="$2"
      checkpoint_seen=1
      shift 2
      ;;
    --checkpoint=*)
      ((checkpoint_seen == 0)) || fail "--checkpoint may only be specified once"
      checkpoint="${1#*=}"
      [[ -n "${checkpoint}" ]] || fail "--checkpoint requires a non-empty value"
      checkpoint_seen=1
      shift
      ;;
    --sim-device)
      ((sim_device_seen == 0)) || fail "--sim-device may only be specified once"
      (($# >= 2)) || fail "--sim-device requires a non-empty value"
      [[ -n "$2" ]] || fail "--sim-device requires a non-empty value"
      validate_sim_device "$2"
      sim_device="$2"
      sim_device_seen=1
      shift 2
      ;;
    --sim-device=*)
      ((sim_device_seen == 0)) || fail "--sim-device may only be specified once"
      sim_device="${1#*=}"
      [[ -n "${sim_device}" ]] || fail "--sim-device requires a non-empty value"
      validate_sim_device "${sim_device}"
      sim_device_seen=1
      shift
      ;;
    -h|--help)
      help_requested=1
      forwarded+=("$1")
      shift
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

if ((checkpoint_seen == 0 && help_requested == 0)); then
  fail "--checkpoint is required"
fi

simpler_root="${project_root}/third_party/SimplerEnv"
maniskill_root="${simpler_root}/ManiSkill2_real2sim"
export PYTHONPATH="${project_root}/src:${simpler_root}:${maniskill_root}"
export MS2_REAL2SIM_ASSET_DIR="${maniskill_root}/data"

command=("${python_executable}" -m qwen3_vl_groot.evaluate_simpler)
if ((checkpoint_seen)); then
  command+=(--checkpoint "${checkpoint}")
fi
command+=(--sim-device "${sim_device}")
command+=("${forwarded[@]}")
exec "${command[@]}"
