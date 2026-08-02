from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Libero10Task:
    name: str
    language_instruction: str


# This is the task order exposed by the pinned official LIBERO-10 benchmark and
# by scripts/evaluate_libero_octo_small.sh. It is also the canonical task_index
# order for the converted libero10_5 training dataset.
LIBERO_10_TASKS = (
    Libero10Task(
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
    Libero10Task(
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "put both the cream cheese box and the butter in the basket",
    ),
    Libero10Task(
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "turn on the stove and put the moka pot on it",
    ),
    Libero10Task(
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
    ),
    Libero10Task(
        (
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_"
            "yellow_and_white_mug_on_the_right_plate"
        ),
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
    ),
    Libero10Task(
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "pick up the book and place it in the back compartment of the caddy",
    ),
    Libero10Task(
        (
            "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_"
            "chocolate_pudding_to_the_right_of_the_plate"
        ),
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    ),
    Libero10Task(
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "put both the alphabet soup and the cream cheese box in the basket",
    ),
    Libero10Task(
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "put both moka pots on the stove",
    ),
    Libero10Task(
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "put the yellow and white mug in the microwave and close it",
    ),
)

LIBERO_10_TASK_NAMES = tuple(task.name for task in LIBERO_10_TASKS)
LIBERO_10_LANGUAGE_INSTRUCTIONS = tuple(
    task.language_instruction for task in LIBERO_10_TASKS
)
LIBERO_10_TASK_COUNT = len(LIBERO_10_TASKS)
LIBERO_10_DEMOS_PER_TASK = 5
