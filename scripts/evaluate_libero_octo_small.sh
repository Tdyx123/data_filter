#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASKS=(
  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
  "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
  "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)

DEFAULT_OUTPUT_ROOT="outputs/octo_small_libero_eval"
SIMULATION_INFRASTRUCTURE_EXIT_CODE=3

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_libero_octo_small.sh [--indexes LIST|all] [EVALUATOR_ARGS...]
  bash scripts/evaluate_libero_octo_small.sh --task-name NAME [EVALUATOR_ARGS...]

Launcher options:
  --indexes LIST     Comma-separated LIBERO-10 indexes, for example 0,2,5.
  --indexes all      Run all ten LIBERO-10 tasks (also the default).
  --output-dir PATH  Result root in indexes mode; each task uses PATH/task-N/.
                     With --task-name, PATH keeps the Python CLI's original meaning.
  --save-videos-path PATH
                     Video root in indexes mode; each task uses PATH/task-N/.
                     With --task-name, videos are written directly under PATH.
  --record-videos K  Save at most K successful and K failed episodes per task.
                     Requires --save-videos-path; defaults to K=1 when omitted.
  --task-name NAME   Run one named task. Cannot be combined with --indexes.
  -h, --help         Show this help and the LIBERO-10 task mapping.

All remaining arguments are forwarded unchanged to:
  python3 -m octo_small_libero.evaluate

Use that module's --help for evaluator options such as --checkpoint, --base-model,
--episodes, --num-envs, --max-steps, --device, --preflight-only, and --smoke-test.
The default protocol runs 150 episodes per task: 50 fixed initial states under
each of the fixed seeds 3471197683, 1232873419, and 1448008435.

LIBERO-10 task mapping:
EOF
  local index
  for index in "${!TASKS[@]}"; do
    printf '  %d  %s\n' "${index}" "${TASKS[${index}]}"
  done
}

fail_usage() {
  printf 'error: %s\n' "$1" >&2
  printf 'Run with --help for usage and the LIBERO-10 task mapping.\n' >&2
  exit 2
}

indexes_explicit=false
indexes_value=""
task_name_explicit=false
task_name=""
output_dir_explicit=false
output_root="${DEFAULT_OUTPUT_ROOT}"
save_videos_path_explicit=false
save_videos_root=""
record_videos_explicit=false
record_videos=""
forwarded_args=()

while (($# > 0)); do
  case "$1" in
    --indexes)
      [[ "${indexes_explicit}" == false ]] || fail_usage "--indexes may only be specified once"
      (($# >= 2)) || fail_usage "--indexes requires a value"
      indexes_explicit=true
      indexes_value="$2"
      shift 2
      ;;
    --indexes=*)
      [[ "${indexes_explicit}" == false ]] || fail_usage "--indexes may only be specified once"
      indexes_explicit=true
      indexes_value="${1#--indexes=}"
      shift
      ;;
    --task-name)
      (($# >= 2)) || fail_usage "--task-name requires a value"
      [[ -n "$2" ]] || fail_usage "--task-name requires a non-empty value"
      task_name_explicit=true
      task_name="$2"
      shift 2
      ;;
    --task-name=*)
      task_name_explicit=true
      task_name="${1#--task-name=}"
      [[ -n "${task_name}" ]] || fail_usage "--task-name requires a non-empty value"
      shift
      ;;
    --output-dir)
      (($# >= 2)) || fail_usage "--output-dir requires a value"
      [[ -n "$2" ]] || fail_usage "--output-dir requires a non-empty value"
      output_dir_explicit=true
      output_root="$2"
      shift 2
      ;;
    --output-dir=*)
      output_dir_explicit=true
      output_root="${1#--output-dir=}"
      [[ -n "${output_root}" ]] || fail_usage "--output-dir requires a non-empty value"
      shift
      ;;
    --save-videos-path)
      [[ "${save_videos_path_explicit}" == false ]] \
        || fail_usage "--save-videos-path may only be specified once"
      (($# >= 2)) || fail_usage "--save-videos-path requires a value"
      [[ -n "$2" ]] || fail_usage "--save-videos-path requires a non-empty value"
      save_videos_path_explicit=true
      save_videos_root="$2"
      shift 2
      ;;
    --save-videos-path=*)
      [[ "${save_videos_path_explicit}" == false ]] \
        || fail_usage "--save-videos-path may only be specified once"
      save_videos_path_explicit=true
      save_videos_root="${1#--save-videos-path=}"
      [[ -n "${save_videos_root}" ]] \
        || fail_usage "--save-videos-path requires a non-empty value"
      shift
      ;;
    --record-videos)
      [[ "${record_videos_explicit}" == false ]] \
        || fail_usage "--record-videos may only be specified once"
      (($# >= 2)) || fail_usage "--record-videos requires a value"
      record_videos_explicit=true
      record_videos="$2"
      shift 2
      ;;
    --record-videos=*)
      [[ "${record_videos_explicit}" == false ]] \
        || fail_usage "--record-videos may only be specified once"
      record_videos_explicit=true
      record_videos="${1#--record-videos=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      forwarded_args+=("$@")
      break
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ "${task_name_explicit}" == true && "${indexes_explicit}" == true ]]; then
  fail_usage "--task-name cannot be combined with --indexes"
fi
if [[ "${record_videos_explicit}" == true && "${save_videos_path_explicit}" == false ]]; then
  fail_usage "--record-videos requires --save-videos-path"
fi
if [[ "${save_videos_path_explicit}" == true ]]; then
  if [[ "${record_videos_explicit}" == false ]]; then
    record_videos="1"
  fi
  if [[ ! "${record_videos}" =~ ^[1-9][0-9]*$ ]]; then
    fail_usage "--record-videos must be a positive integer"
  fi
fi

if [[ "${task_name_explicit}" == true ]]; then
  single_task_args=("${forwarded_args[@]}" --task-name "${task_name}")
  if [[ "${output_dir_explicit}" == true ]]; then
    single_task_args+=(--output-dir "${output_root}")
  fi
  if [[ "${save_videos_path_explicit}" == true ]]; then
    single_task_args+=(
      --save-videos-path "${save_videos_root}"
      --record-videos "${record_videos}"
    )
  fi
  exec python3 -m octo_small_libero.evaluate "${single_task_args[@]}"
fi

selected_indexes=()
if [[ "${indexes_explicit}" == false || "${indexes_value}" == "all" ]]; then
  selected_indexes=(0 1 2 3 4 5 6 7 8 9)
else
  [[ -n "${indexes_value}" ]] || fail_usage "--indexes requires a non-empty value"
  if [[ ! "${indexes_value}" =~ ^[0-9](,[0-9])*$ ]]; then
    fail_usage "--indexes must be 'all' or a comma-separated list of integers from 0 to 9"
  fi

  declare -A seen_indexes=()
  IFS=',' read -r -a requested_indexes <<<"${indexes_value}"
  for index in "${requested_indexes[@]}"; do
    if [[ -n "${seen_indexes[${index}]+present}" ]]; then
      fail_usage "--indexes contains duplicate index ${index}"
    fi
    seen_indexes["${index}"]=true
    selected_indexes+=("${index}")
  done
fi

failed_indexes=()
failed_statuses=()
for index in "${selected_indexes[@]}"; do
  task_name="${TASKS[${index}]}"
  task_output_dir="${output_root%/}/task-${index}"
  [[ -n "${output_root%/}" ]] || task_output_dir="/task-${index}"
  printf '[info] evaluating LIBERO-10 task %d: %s\n' "${index}" "${task_name}"
  printf '[info] output directory: %s\n' "${task_output_dir}"

  task_args=(
    "${forwarded_args[@]}"
    --task-name "${task_name}"
    --output-dir "${task_output_dir}"
  )
  if [[ "${save_videos_path_explicit}" == true ]]; then
    task_video_dir="${save_videos_root%/}/task-${index}"
    [[ -n "${save_videos_root%/}" ]] || task_video_dir="/task-${index}"
    printf '[info] video directory: %s\n' "${task_video_dir}"
    task_args+=(
      --save-videos-path "${task_video_dir}"
      --record-videos "${record_videos}"
    )
  fi
  if python3 -m octo_small_libero.evaluate "${task_args[@]}"; then
    printf '[info] task %d completed successfully\n' "${index}"
  else
    status=$?
    if ((status == SIMULATION_INFRASTRUCTURE_EXIT_CODE)); then
      printf '[error] task %d hit a simulation infrastructure failure; aborting the batch\n' \
        "${index}" >&2
      exit "${SIMULATION_INFRASTRUCTURE_EXIT_CODE}"
    fi
    failed_indexes+=("${index}")
    failed_statuses+=("${status}")
    printf '[error] task %d failed with exit code %d; continuing\n' \
      "${index}" "${status}" >&2
  fi
done

if ((${#failed_indexes[@]} > 0)); then
  printf '[error] failed LIBERO-10 task indexes:' >&2
  for position in "${!failed_indexes[@]}"; do
    printf ' %s(exit=%s)' \
      "${failed_indexes[${position}]}" "${failed_statuses[${position}]}" >&2
  done
  printf '\n' >&2
  exit 1
fi

printf '[info] all selected LIBERO-10 tasks completed successfully\n'
