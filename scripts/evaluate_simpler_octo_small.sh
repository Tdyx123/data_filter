#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="python3"
python_seen=0
checkpoint=""
checkpoint_seen=0
base_model=""
base_model_seen=0
statistics=""
statistics_seen=0
help_requested=0
forwarded=()

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 2
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
    --base-model)
      ((base_model_seen == 0)) || fail "--base-model may only be specified once"
      (($# >= 2)) || fail "--base-model requires a non-empty value"
      [[ -n "$2" ]] || fail "--base-model requires a non-empty value"
      base_model="$2"
      base_model_seen=1
      shift 2
      ;;
    --base-model=*)
      ((base_model_seen == 0)) || fail "--base-model may only be specified once"
      base_model="${1#*=}"
      [[ -n "${base_model}" ]] || fail "--base-model requires a non-empty value"
      base_model_seen=1
      shift
      ;;
    --statistics)
      ((statistics_seen == 0)) || fail "--statistics may only be specified once"
      (($# >= 2)) || fail "--statistics requires a non-empty value"
      [[ -n "$2" ]] || fail "--statistics requires a non-empty value"
      statistics="$2"
      statistics_seen=1
      shift 2
      ;;
    --statistics=*)
      ((statistics_seen == 0)) || fail "--statistics may only be specified once"
      statistics="${1#*=}"
      [[ -n "${statistics}" ]] || fail "--statistics requires a non-empty value"
      statistics_seen=1
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

if ((help_requested == 0)); then
  ((checkpoint_seen)) || fail "--checkpoint is required"
  ((base_model_seen)) || fail "--base-model is required"
  ((statistics_seen)) || fail "--statistics is required"
fi

simpler_root="${project_root}/third_party/SimplerEnv"
maniskill_root="${simpler_root}/ManiSkill2_real2sim"
export PYTHONPATH="${project_root}/src:${simpler_root}:${maniskill_root}"
export MS2_REAL2SIM_ASSET_DIR="${maniskill_root}/data"

command=("${python_executable}" -m octo_small_bridge.evaluate_simpler)
if ((checkpoint_seen)); then
  command+=(--checkpoint "${checkpoint}")
fi
if ((base_model_seen)); then
  command+=(--base-model "${base_model}")
fi
if ((statistics_seen)); then
  command+=(--statistics "${statistics}")
fi
command+=("${forwarded[@]}")
exec "${command[@]}"
