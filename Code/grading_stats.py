import ast
from enum import Enum
from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from Levenshtein import distance, ratio

from quiz_generation import GROUNDINGS, SEPARATOR


class FailureMode(Enum):
    COPY_CONTEXT = 0
    SCRAMBLED = 1
    WRONG_COMBINATION = 2
    OTHER = 3


def extract_content(response: str, first_and_last: bool = False) -> str | List[str]:
    """
    Extracts and returns the first contentful line in the response.
    """
    extracted_list = [s for s in response.replace(SEPARATOR, "\n").split("\n") if s]
    if extracted_list:
        extracted = (
            [extracted_list[0], extracted_list[-1]]
            if first_and_last
            else extracted_list[0]
        )
    else:
        extracted = [] if first_and_last else ""
    return extracted


def is_correct(response: str, answer: str, first_and_last: bool = False) -> bool:
    """
    Checks if the response matches the answer. Specifically,
    it checks the first contentful line in the response and
    compares it to the answer when both have all whitespace removed.
    """
    if first_and_last:
        for candidate in extract_content(response, first_and_last):
            if "".join(candidate.split()) == "".join(answer.split()):
                return True
        return False
    else:
        return "".join(extract_content(response).split()) == "".join(answer.split())


def indel_ratio(response: str, answer: str) -> float:
    """
    Returns the normalized InDel ratio of the response and answer after being preprocessed
    with extract_content and removal of whitespace.
    """
    return ratio(
        response, answer, processor=lambda s: "".join(extract_content(s).split())
    )


def is_scrambled(response: str, answer: str) -> bool:
    """
    Returns True if response is an anagram of answer, ignoring whitespace,
    and returns False otherwise. Uses same filtering as is_correct to
    extract responses from response string.
    """
    return sorted("".join(extract_content(response).split())) == sorted(
        "".join(answer.split())
    )


def grounding_context(grounding: str) -> List[str]:
    """
    Takes one of the 16 groundings used in the pilot and returns a list of the
    3 other groundings associated with it (which would have been presented in a
    question which has the inputted grounding as an answer).
    """
    for grounding_list in GROUNDINGS:
        if grounding in grounding_list:
            return [g for g in grounding_list.flatten() if g != grounding]
    raise ValueError(f"Grounding {grounding} not recognized.")


def get_failure_mode(response: str, answer: str) -> FailureMode:
    """
    Takes an incorrect response and the corresponding correct answer
    and returns a FailureMode describing the way in which the response
    is incorrect.
    """
    context = grounding_context(answer)  #! This doesn't extend to phase 2
    for grounding in context:
        if is_correct(response, grounding):
            return FailureMode.COPY_CONTEXT
    if is_scrambled(response, answer):
        return FailureMode.SCRAMBLED
    if set("".join(extract_content(response).split())).issubset(set("".join(context))):
        return FailureMode.WRONG_COMBINATION
    return FailureMode.OTHER


def grade_charitably(
    response: str,
    answer: str,
    pairs: List[Tuple[str, str]],
    prompt: str = None,
    case_sensitive: bool = False,
    grading_fxn: Callable[[str, str], bool] = is_correct,
    first_and_last: bool = False,
) -> bool:
    """
    Grades a response relative to a correct answer given a grading function
    and a list of pairs of characters that could have been confused with one another.
    If the responses is correct with some combination of replacements of the pairs,
    the response is graded as correct (True). The response is graded incorrect otherwise.
    """
    if prompt:
        if response.startswith(prompt):
            response = response[len(prompt) :]
    if not case_sensitive:
        response, answer = response.upper(), answer.upper()
    if grading_fxn(response, answer, first_and_last):
        return True
    elif len(pairs) == 0:
        return False
    else:
        return (
            grade_charitably(
                response.replace(*pairs[0]),
                answer,
                pairs[1:],
                case_sensitive=case_sensitive,
                first_and_last=first_and_last,
            )
            or grade_charitably(
                response.replace(*pairs[0][::-1]),
                answer,
                pairs[1:],
                case_sensitive=case_sensitive,
                first_and_last=first_and_last,
            )
            or grade_charitably(
                response,
                answer,
                pairs[1:],
                case_sensitive=case_sensitive,
                first_and_last=first_and_last,
            )
        )


def average_propagate_stds(
    data: List[float], stds: List[float], stderrs: List[float]
) -> tuple[float, float, float]:
    """
    Takes a list of data points, their standard deviations, and their standard errors and
    returns the mean of the data with propagated standard deviation and standard error.
    """
    data_np = np.asarray(data)
    stds_np = np.asarray(stds)
    stderrs_np = np.asarray(stderrs)

    mean = np.mean(data_np)
    std = np.divide(np.sqrt(np.sum(np.square(stds_np))), len(stds))
    stderr = np.divide(np.sqrt(np.sum(np.square(stderrs_np))), len(stderrs))

    return mean, std, stderr


def summarize_participant_results(results: dict[str, list[float]]) -> tuple[float]:
    """
    Summarizes results from a participant (formatted as the condition_stats dictionary from score_stats)
    by averaging the average scores from each condition with error propagation.
    """
    return average_propagate_stds(
        *list(
            map(
                results.get,
                [
                    "avg_condition_accuracy",
                    "stdevs_per_condition",
                    "stderrs_per_condition",
                ],
            )
        )
    )


def score_stats(
    avg_grade_per_q: List[Number],
    avg_grade_per_q_stds: List[Number],
    avg_grade_per_q_stderrs: List[Number],
    experiment_conditions: List[str],
    questions_per_quiz: int,
    model_name=None,
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Create and return two dictionaries containing stats based on
    the scoring information in avg_grade_per_q, the names of
    the experiment conditions, and the number of questions in a quiz.
    """
    condition_stats = {
        "avg_condition_accuracy": [],
        "stdevs_per_condition": [],
        "stderrs_per_condition": [],
    }
    question_stats = {
        "regressions": [],
        "avg_q_accuracy": [],
        "stdevs_per_q_num": [],
        "stderrs_per_q_num": [],
    }

    for experiment_name in experiment_conditions:
        if experiment_name in [
            "random_permuted_pairs",
            "randoms",
            "categorial",
            "multi_attribute",
            "numeric",
            "numeric_multi_attribute",
            "relational",
        ]:
            num_quizzes = 2
        else:
            num_quizzes = 4
        y = np.array(avg_grade_per_q[: questions_per_quiz * num_quizzes])

        y_stds = np.array(avg_grade_per_q_stds[: questions_per_quiz * num_quizzes])

        y_stderrs = np.array(
            avg_grade_per_q_stderrs[: questions_per_quiz * num_quizzes]
        )

        # y is the score per question in this condition, and the associated std per question (treating a q num in diff quizes as a diff question)

        means_per_q_num = []
        stds_per_q_num = []
        stderrs_per_q_num = []
        for i in range(questions_per_quiz):
            mean, std, stderr = average_propagate_stds(
                y[i::questions_per_quiz].copy(),
                y_stds[i::questions_per_quiz].copy(),
                y_stderrs[i::questions_per_quiz].copy(),
            )
            means_per_q_num.append(mean)
            stds_per_q_num.append(std)
            stderrs_per_q_num.append(stderr)

        question_stats["regressions"].append(
            scipy.stats.linregress(np.arange(len(means_per_q_num)), means_per_q_num)
        )
        question_stats["avg_q_accuracy"].append(means_per_q_num)
        question_stats["stdevs_per_q_num"].append(stds_per_q_num)
        question_stats["stderrs_per_q_num"].append(stderrs_per_q_num)

        condition_accuracy, condition_std, condition_stderr = average_propagate_stds(
            means_per_q_num, stds_per_q_num, stderrs_per_q_num
        )
        condition_stats["avg_condition_accuracy"].append(condition_accuracy)
        condition_stats["stdevs_per_condition"].append(condition_std)
        condition_stats["stderrs_per_condition"].append(condition_stderr)
        avg_grade_per_q = avg_grade_per_q[questions_per_quiz * num_quizzes :]
        avg_grade_per_q_stds = avg_grade_per_q_stds[questions_per_quiz * num_quizzes :]
        avg_grade_per_q_stderrs = avg_grade_per_q_stderrs[
            questions_per_quiz * num_quizzes :
        ]
    return condition_stats, question_stats


def model_stats(
    paths: List[str],
    answer_key: List[Tuple[str, Optional[str], bool]],
    experiment_conditions: List[str],
    num_quizzes: int,
    questions_per_quiz: int,
    samples_per_q: int,
    model_name: str,
    all_subjects_df,
    suppress_get_failure_modes=False,
    first_and_last: bool = False,
) -> Tuple[
    Dict[str, List], Dict[str, List], List[List[Tuple[int, str, str, str]]], List[int]
]:
    """
    Takes a list of paths to a text file of a model's responses to quizzes
    as well as an answer key, the names of the experiment conditions,
    the number of questions per quiz and the number of samples taken
    per question and returns dictionaries with stats for the model.
    """

    responses = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            responses += ast.literal_eval(f.read().split("printing all_responses")[1])

    model_grades = []
    incorrect_responses: List[List[Tuple[int, str, str, str]]] = []
    failure_modes: List[int] = [0 for _ in range(len(FailureMode))]
    # scores (all 0 or 1) for each sampled response
    for quiz in range(num_quizzes):
        wrongs = []
        for question in range(questions_per_quiz):
            for sample in range(samples_per_q):
                response_index = (
                    quiz * questions_per_quiz * samples_per_q
                    + question * samples_per_q
                    + sample
                )
                score = int(
                    grade_charitably(
                        responses[response_index],
                        answer_key[response_index // samples_per_q][0],
                        [("|", "I")],
                        prompt=answer_key[response_index // samples_per_q][1],
                        case_sensitive=answer_key[response_index // samples_per_q][2],
                        first_and_last=first_and_last,
                    )
                )
                model_grades.append(score)

                df_app = pd.DataFrame(
                    {
                        "subject_type": model_name,
                        "quiz_number": quiz,
                        "quiz_class": "",
                        "respondent_scores": score,
                        "question_num": (question + 1),
                    },
                    index=[0],
                )
                all_subjects_df = pd.concat(
                    [all_subjects_df, df_app], ignore_index=True
                )

                if score != 1:
                    prev_answer = " "
                    if ((response_index - samples_per_q) // samples_per_q) >= 0:
                        prev_answer = answer_key[
                            (response_index - samples_per_q) // samples_per_q
                        ][0]
                    wrongs.append(
                        (
                            question,
                            responses[response_index],
                            answer_key[response_index // samples_per_q][0],
                            prev_answer,
                        )
                    )
                    if not (suppress_get_failure_modes):
                        failure_modes[
                            get_failure_mode(
                                responses[response_index],
                                answer_key[response_index // samples_per_q][0],
                            ).value
                        ] += 1
        incorrect_responses.append(wrongs)

    if not (suppress_get_failure_modes):
        failure_modes = [n / sum(failure_modes) for n in failure_modes]

    # average score of sampled responses for each question

    model_avg_grade_per_q = [
        np.average(model_grades[i : i + samples_per_q])
        for i in range(0, len(model_grades), samples_per_q)
    ]

    model_avg_grade_per_q_stds = [
        np.std(model_grades[i : i + samples_per_q])
        for i in range(0, len(model_grades), samples_per_q)
    ]

    model_avg_grade_per_q_stderrs = [
        (
            np.std(model_grades[i : i + samples_per_q])
            / np.sqrt(np.size(model_grades[i : i + samples_per_q]))
        )
        for i in range(0, len(model_grades), samples_per_q)
    ]

    return (
        *score_stats(
            model_avg_grade_per_q,
            model_avg_grade_per_q_stds,
            model_avg_grade_per_q_stderrs,
            experiment_conditions,
            questions_per_quiz,
            model_name=model_name,
        ),
        incorrect_responses,
        failure_modes,
        all_subjects_df,
    )


PILOT_ANSWERS: List[Tuple[str, Optional[str], bool]] = [
    [
        ("C % K % E", "oval => ", False),
        ("c c", "woman => ", True),
        ("*", "pink => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "kitten => ", False),
        ("*", "woman => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "pink => ", False),
        ("C % K % E", "kitten => ", False),
        ("c c", "oval => ", True),
        ("C % K % E", "woman => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "oval => ", False),
        ("c c", "pink => ", True),
        ("*", "kitten => ", False),
        ("C % K % E", "pink => ", False),
        ("c c", "kitten => ", True),
        ("Q Q Z Z I I Q Q Z Z I I", "woman => ", False),
        ("*", "oval => ", False),
    ],
    [
        ("C % K % E", "oval => ", False),
        ("c c", "woman => ", True),
        ("*", "pink => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "kitten => ", False),
        ("*", "woman => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "pink => ", False),
        ("C % K % E", "kitten => ", False),
        ("c c", "oval => ", True),
        ("C % K % E", "woman => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "oval => ", False),
        ("c c", "pink => ", True),
        ("*", "kitten => ", False),
        ("C % K % E", "pink => ", False),
        ("c c", "kitten => ", True),
        ("Q Q Z Z I I Q Q Z Z I I", "woman => ", False),
        ("*", "oval => ", False),
    ],
    [
        ("C K E", "circle => ", False),
        ("c c", "woman => ", True),
        ("} * {", "red => ", False),
        ("Q Z I", "dog => ", False),
        ("*", "woman => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "pink => ", False),
        ("E K C", "dog => ", False),
        ("C C C", "square => ", True),
        ("E K C", "king => ", False),
        ("Q Z I Q Z I", "circle => ", False),
        ("c c", "pink => ", True),
        ("*", "kitten => ", False),
        ("C % K % E", "pink => ", False),
        ("c c", "kitten => ", True),
        ("Q Q Z Z I I", "queen => ", False),
        ("} * {", "circle => ", False),
    ],
    [
        ("c c", "woman => ", True),
        ("*", "pink => ", False),
        ("C % K % E", "oval => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "kitten => ", False),
        ("*", "woman => ", False),
        ("C % K % E", "kitten => ", False),
        ("c c", "oval => ", True),
        ("Q Q Z Z I I Q Q Z Z I I", "pink => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "oval => ", False),
        ("C % K % E", "woman => ", False),
        ("*", "kitten => ", False),
        ("c c", "pink => ", True),
        ("*", "oval => ", False),
        ("C % K % E", "pink => ", False),
        ("c c", "kitten => ", True),
        ("Q Q Z Z I I Q Q Z Z I I", "woman => ", False),
    ],
    [
        ("E K C", "zebra => ", False),
        ("Q Z I Q Z I", "bean => ", False),
        ("c c", "wife => ", True),
        ("*", "carpet => ", False),
        ("C % K % E", "wife => ", False),
        ("c c", "carpet => ", True),
        ("Q Q Z Z I I", "bullet => ", False),
        ("} * {", "bean => ", False),
    ],
    [
        ("C % K % E", "plug => ", False),
        ("c c", "book => ", True),
        ("*", "wife => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "carpet => ", False),
        ("*", "book => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "wife => ", False),
        ("C % K % E", "carpet => ", False),
        ("c c", "plug => ", True),
    ],
]

FOLLOWUP_ANSWERS: List[Tuple[str, Optional[str], bool]] = [
    [
        ("C % K % E", None, False),
        ("c c", None, True),
        ("*", None, False),
        ("Q Q Z Z I I Q Q Z Z I I", None, False),
        ("*", None, False),
        ("Q Q Z Z I I Q Q Z Z I I", None, False),
        ("C % K % E", None, False),
        ("c c", None, True),
        ("C % K % E", None, False),
        ("Q Q Z Z I I Q Q Z Z I I", None, False),
        ("c c", None, True),
        ("*", None, False),
        ("C % K % E", None, False),
        ("c c", None, True),
        ("Q Q Z Z I I Q Q Z Z I I", None, False),
        ("*", None, False),
    ],
    [
        ("C % K % E", "lime => ", False),
        ("c c", "tree => ", True),
        ("*", "door => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "pillow => ", False),
        ("*", "pillow => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "door => ", False),
        ("C % K % E", "lime => ", False),
        ("c c", "tree => ", True),
        ("C % K % E", "tree => ", False),
        ("Q Q Z Z I I Q Q Z Z I I", "lime => ", False),
        ("c c", "pillow => ", True),
        ("*", "door => ", False),
        ("C % K % E", "door => ", False),
        ("c c", "lime => ", True),
        ("Q Q Z Z I I Q Q Z Z I I", "pillow => ", False),
        ("*", "tree => ", False),
    ],
]

PHASE_2_ANSWERS: List[Tuple[str, Optional[str], bool]] = [
    [
        ("*", "human => ", False),
        ("!", "unicycle => ", False),
        ("!", "day => ", True),
        ("*", "power outlet => ", False),
        ("*", "dollar => ", False),
        ("!", "bicycle => ", True),
        ("*", "donut => ", False),
        ("*", "human => ", False),
    ],
    [
        ("* * *", "sister => ", False),
        ("* *", "square => ", True),
        ("[ ! ]", "microphone => ", False),
        ("[ * ]", "golf => ", False),
        ("[ ! ]", "chess => ", False),
        ("! !", "circle => ", False),
        ("* * *", "sister => ", False),
        ("( * )", "screen => ", True),
    ],
    [
        ("* * * * * * *", "day => ", False),
        ("*", "unicycle => ", False),
        ("* * * * * *", "bee => ", True),
        ("*", "headband => ", False),
        ("* * * * * *", "bee => ", False),
        ("* * * *", "car => ", True),
        ("* * * *", "t-shirt => ", False),
        ("*", "week => ", False),
    ],
    [
        ("* *", "human => ", False),
        ("! !", "bicycle => ", True),
        ("&", "feet => ", False),
        ("! ! ! !", "t-shirt => ", False),
        ("* * *", "power outlet => ", False),
        ("!", "week => ", False),
        ("! !", "chicken => ", False),
        ("!", "unicycle => ", True),
    ],
    [
        ("M # M", "hat => ", False),
        ("Y", "axe => ", True),
        ("C ; ;", "south => ", False),
        ("P", "cow => ", False),
        ("K", "rake => ", False),
        ("C C C", "west => ", False),
        ("P", "dog => ", False),
        ("[ K ]", "pants => ", True),
    ],
]
