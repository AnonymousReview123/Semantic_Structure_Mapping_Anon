import math
from itertools import cycle
from numbers import Number
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats.distributions import chi2

import grading_stats
import plotting

sns.set_palette("colorblind")

experiment_conditions = [
    "defaults",  # 0-3
    "distracted",  # 4-7
    "permuted_pairs",  # 8-11
    "permuted_questions",  # 12-15
    "random_permuted_pairs",  # 16,17
    "randoms",  # 18,19
    "only_rhs",  # 20-23
    "random_finals",  # 24-27
]


def quiz_to_condition(n: Number) -> str:
    n = int(n)
    if 0 <= n <= 15:
        return experiment_conditions[n // 4]
    elif 16 <= n <= 19:
        return experiment_conditions[4 + ((n - 16) // 2)]
    else:
        return experiment_conditions[6 + ((n - 20) // 4)]


def overall_number_to_quiz_number(n: Number) -> int:
    """
    Takes a quiz number from 0 to 28 and returns the quiz number from 1 to 4.
    """
    n = int(n)
    if 0 <= n <= 17 or 20 <= n <= 27:
        return (n % 4) + 1
    elif n in [18, 19]:
        return (n % 4) - 1
    else:
        raise ValueError(
            f"Invalid quiz number {n}. Number must be an int from 0 to 27."
        )


def quiz_to_quiz_modded(n: Number) -> Number:
    n = int(n)
    if 0 <= n <= 15:
        return (n % 4) + 1
    elif 16 <= n <= 19:
        return ((n - 16) % 2) + 1
    else:
        return ((n - 20) % 4) + 1


def classify_quiz(row: pd.Series):
    quiz_num = row["quiz_number"]
    if math.isnan(quiz_num):
        return "NA"
    return quiz_to_condition(quiz_num)


num_quizzes = 28
questions_per_quiz = 4
samples_per_q = 5

# answer key as a flat list
answer_key: List[Tuple[str, bool]] = [
    answer
    for condition in grading_stats.PILOT_ANSWERS + grading_stats.FOLLOWUP_ANSWERS
    for answer in condition
]

all_subjects_df = pd.DataFrame(
    columns=[
        "subject_type",
        "quiz_number",
        "quiz_class",
        "respondent_scores",
        "question_num",
    ]
)

(
    gpt3_arrows_condition_stats,
    gpt3_arrows_question_stats,
    gpt3_arrows_incorrect_responses,
    gpt3_arrows_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_1/GPT_arrows/GPT-3/gpt-3_results_arrows.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="GPT-3",
    all_subjects_df=all_subjects_df,
)


(
    gpt4_arrows_condition_stats,
    gpt4_arrows_question_stats,
    gpt4_arrows_incorrect_responses,
    gpt4_arrows_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_1/GPT_arrows/GPT-4/gpt-4_results_arrows.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="GPT-4",
    all_subjects_df=all_subjects_df,
)


(
    pythia12b_deduped_condition_stats,
    pythia12b_deduped_question_stats,
    pythia12b_deduped_incorrect_responses,
    pythia12b_deduped_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_1/Pythia/12b-deduped_results.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Pythia-12B-Deduped",
    all_subjects_df=all_subjects_df,
)

(
    falcon40b_arrows_condition_stats,
    falcon40b_arrows_question_stats,
    falcon40b_arrows_responses,
    falcon40b_arrows_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_1/Falcon_arrows/falcon_40b_results_arrows.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Falcon-40B-arrows",
    all_subjects_df=all_subjects_df,
)

(
    claude2_condition_stats,
    claude2_question_stats,
    claude2_incorrect_responses,
    claude2_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_1/Claude/Claude 2/claude-2_long_results.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Claude-2",
    all_subjects_df=all_subjects_df,
    first_and_last=True,
)

(
    claude3opus_condition_stats,
    claude3opus_question_stats,
    claude3opus_incorrect_responses,
    claude3opus_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_1/Claude/Claude 3 Opus/claude-3-opus-20240229_results.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Claude-3-Opus",
    all_subjects_df=all_subjects_df,
    first_and_last=True,
)

all_subjects_df["quiz_class"] = all_subjects_df.apply(classify_quiz, axis=1)

pilot_df = pd.read_csv(
    "quiz_results/phase_1/Human/Final+pilot+survey_May+1,+2023_12.26/Final pilot survey_June 1, 2023_12.35.csv"
)

followup_df = pd.read_csv(
    "quiz_results/phase_1/Human/Pilot+survey+follow+up+controls_May+17,+2023_11.58/Pilot survey follow up controls_June 1, 2023_12.39.csv"
)
followup_q_columns = [
    [f"Q{n}{s}" for n in [1, 3, 5, 7, 9, 10, 11]]
    for s in [""] + [f".{i}" for i in range(1, 8)]
]
followup_column_mapping = {
    q: f"{q.split('.')[0]}.{i+20}" for i in range(8) for q in followup_q_columns[i]
}
followup_df.rename(columns=followup_column_mapping, inplace=True)

pilot_df = pilot_df.rename(
    columns={"Q226": "Consent", "Q227": "Prolific_ID", "Q222": "Attention"}
)

followup_df = followup_df.rename(
    columns={"Q90": "Consent", "Q91": "Prolific_ID", "Q92": "Attention"}
)

human_df = pd.concat([pilot_df, followup_df], axis=0, ignore_index=True)


human_df.insert(0, "PROLIFIC_PID", human_df.pop("PROLIFIC_PID"))
human_df.insert(1, "Prolific_ID", human_df.pop("Prolific_ID"))

in_person_df = pd.read_csv(
    "quiz_results/phase_1/Human/In_Person_Phase_1/Phase+1+(Students)_October+8,+2023_15.09/Phase 1 (Students)_October 15, 2023_10.49.csv"
)

in_person_df["RecipientEmail"] = in_person_df["Q347"]
in_person_df = in_person_df.drop(columns=["Q347"])
in_person_df["RecipientEmail"] = in_person_df["RecipientEmail"].replace(np.nan, "")
in_person_df = in_person_df[in_person_df["RecipientEmail"].str.contains("@University_Name.edu")]
in_person_df = in_person_df.drop_duplicates(subset=["RecipientEmail"], keep="last")

in_person_df = in_person_df.rename(columns={"Q226": "Consent", "Q222": "Attention"})

in_person_df.insert(0, "PROLIFIC_PID", "")
in_person_df.insert(1, "Prolific_ID", "")

human_df.drop(
    human_df.index[[0, 1]], inplace=True
)

human_df = pd.concat([human_df, in_person_df], axis=0, ignore_index=True)


def classify_in_person(row: pd.Series) -> int:
    if len(row["Prolific_ID"]) > 0:
        return False
    else:
        return True


human_df["In_Person"] = human_df.apply(classify_in_person, axis=1)

def extract_quiz_number(row: pd.Series) -> int:
    if row["Finished"].lower() != "true":
        return None
    for i in range(28):
        column = "Q1"
        if i >= 1:
            column += f".{i}"
        if pd.notnull(row[column]):
            return i


human_df["quiz_number"] = human_df.apply(extract_quiz_number, axis=1)


def classify_quiz(row: pd.Series):
    quiz_num = row["quiz_number"]
    if math.isnan(quiz_num):
        return "NA"
    return quiz_to_condition(quiz_num)


human_df["quiz_class"] = human_df.apply(classify_quiz, axis=1)

human_df["duration_float"] = pd.to_numeric(
    human_df["Duration (in seconds)"], errors="coerce"
)

print(
    "Num human subjects phase 1:", human_df.loc[human_df["In_Person"] == True].shape[0]
)

print("Grouped by condition:")

human_df_for_counts = human_df.loc[human_df["In_Person"] == True]

print(human_df_for_counts.groupby(['quiz_class']).size())

mn = human_df["duration_float"].mean()

std = human_df["duration_float"].std()

human_df = human_df[(human_df["duration_float"]) - mn <= 2 * std]

human_df = human_df[mn - (human_df["duration_float"]) <= 2 * std]

human_df = human_df[human_df["Attention"] == "8"]


def score_respondent(
    row: pd.Series,
    q_scores_per_quiz: List[List[List]],
    incorrect_responses: List[Tuple[str, str]],
    individual_scores: List[List],
    failure_modes: List[int],
    scores_grouped_by_quiz: List[List[List]],
):
    this_respondent_scores = []
    quiz_number = row["quiz_number"]
    if pd.isna(quiz_number):
        return
    else:
        quiz_number = int(quiz_number)
    questions = ["Q1", "Q3", "Q5", "Q7"]
    if quiz_number >= 1:
        questions = [f"{q}.{quiz_number}" for q in questions]
    responses = [row[q] for q in questions]
    answers = answer_key[4 * quiz_number : 4 * quiz_number + 4]
    prev_answers = answer_key[max(0, 4 * quiz_number - 1) : 4 * quiz_number + 4 - 1]
    if len(prev_answers) < len(
        answers
    ):  # max kicked in because we were at the beginning
        prev_answers.insert(0, " ")
    for i, (response, answer, prev_answer) in enumerate(
        zip(responses, answers, prev_answers)
    ):
        score = int(
            grading_stats.grade_charitably(
                response,
                answer[0],
                [("|", "I")],
                prompt=answer[1],
                case_sensitive=answer[2],
            )
        )
        q_scores_per_quiz[quiz_number][i].append(score)
        this_respondent_scores.append(score)
        if score != 1:
            incorrect_responses.append((response, answer[0], prev_answer[0]))
            failure_modes[
                grading_stats.get_failure_mode(response, answer[0]).value
            ] += 1
    respondent_overall_score = np.mean(np.asarray(this_respondent_scores))
    individual_scores[experiment_conditions.index(row["quiz_class"])].append(
        respondent_overall_score
    )
    scores_grouped_by_quiz[overall_number_to_quiz_number(row["quiz_number"]) - 1][
        experiment_conditions.index(row["quiz_class"])
    ].append(respondent_overall_score)
    return respondent_overall_score


def score_respondent_all(
    row: pd.Series,
    q_scores_per_quiz: List[List[List]],
    incorrect_responses: List[Tuple[str, str]],
    failure_modes: List[int],
):
    this_respondent_scores = []
    quiz_number = row["quiz_number"]
    if pd.isna(quiz_number):
        return
    else:
        quiz_number = int(quiz_number)
    questions = ["Q1", "Q3", "Q5", "Q7"]
    if quiz_number >= 1:
        questions = [f"{q}.{quiz_number}" for q in questions]
    responses = [row[q] for q in questions]
    answers = answer_key[4 * quiz_number : 4 * quiz_number + 4]
    prev_answers = answer_key[max(0, 4 * quiz_number - 1) : 4 * quiz_number + 4 - 1]
    if len(prev_answers) < len(
        answers
    ):  # max kicked in because we were at the beginning
        prev_answers.insert(0, " ")
    for i, (response, answer, prev_answer) in enumerate(
        zip(responses, answers, prev_answers)
    ):
        score = int(
            grading_stats.grade_charitably(
                response,
                answer[0],
                [("|", "I")],
                prompt=answer[1],
                case_sensitive=answer[2],
            )
        )
        q_scores_per_quiz[quiz_number][i].append(score)
        this_respondent_scores.append(score)
        if score != 1:
            incorrect_responses.append((response, answer[0], prev_answer[0]))
            failure_modes[
                grading_stats.get_failure_mode(response, answer[0]).value
            ] += 1
    this_respondent_scores_arr = np.asarray(this_respondent_scores)
    return this_respondent_scores_arr


prolific_df = human_df.loc[human_df["In_Person"] == False].copy()
University_Name_df = human_df.loc[human_df["In_Person"] == True].copy()

prolific_q_scores_per_quiz = [[[] for _ in range(4)] for _ in range(28)]
prolific_incorrect_responses: List[Tuple[str, str]] = []
prolific_failure_modes: List[int] = [0 for _ in range(len(grading_stats.FailureMode))]

prolific_individual_scores = [[] for _ in range(len(experiment_conditions))]
prolific_scores_grouped_by_quiz = [
    [[] for _ in range(len(experiment_conditions))] for _ in range(4)
]

prolific_df["respondent_score"] = prolific_df.apply(
    lambda row: score_respondent(
        row,
        prolific_q_scores_per_quiz,
        prolific_incorrect_responses,
        prolific_individual_scores,
        prolific_failure_modes,
        prolific_scores_grouped_by_quiz,
    ),
    axis=1,
)
prolific_failure_modes = [
    n / sum(prolific_failure_modes) for n in prolific_failure_modes
]

University_Name_q_scores_per_quiz = [[[] for _ in range(4)] for _ in range(28)]
University_Name_incorrect_responses: List[Tuple[str, str]] = []
University_Name_failure_modes: List[int] = [0 for _ in range(len(grading_stats.FailureMode))]

University_Name_individual_scores = [[] for _ in range(len(experiment_conditions))]
University_Name_scores_grouped_by_quiz = [
    [[] for _ in range(len(experiment_conditions))] for _ in range(4)
]

University_Name_df["respondent_score"] = University_Name_df.apply(
    lambda row: score_respondent(
        row,
        University_Name_q_scores_per_quiz,
        University_Name_incorrect_responses,
        University_Name_individual_scores,
        University_Name_failure_modes,
        University_Name_scores_grouped_by_quiz,
    ),
    axis=1,
)
University_Name_failure_modes = [
    n / sum(University_Name_failure_modes) for n in University_Name_failure_modes
]

University_Name_failure_modes = [
    n / sum(University_Name_failure_modes) for n in University_Name_failure_modes
]

prolific_avg_grade_per_q = [
    np.average(q_scores) for quiz in prolific_q_scores_per_quiz for q_scores in quiz
]

prolific_avg_grade_per_q_stds = [
    np.std(q_scores) for quiz in prolific_q_scores_per_quiz for q_scores in quiz
]

prolific_avg_grade_per_q_stderrs = [
    (np.std(q_scores) / np.sqrt(np.size(q_scores)))
    for quiz in prolific_q_scores_per_quiz
    for q_scores in quiz
]

prolific_condition_stats, prolific_question_stats = grading_stats.score_stats(
    prolific_avg_grade_per_q,
    prolific_avg_grade_per_q_stds,
    prolific_avg_grade_per_q_stderrs,
    experiment_conditions,
    questions_per_quiz,
)

University_Name_avg_grade_per_q = [
    np.average(q_scores) for quiz in University_Name_q_scores_per_quiz for q_scores in quiz
]

University_Name_avg_grade_per_q_stds = [
    np.std(q_scores) for quiz in University_Name_q_scores_per_quiz for q_scores in quiz
]

University_Name_avg_grade_per_q_stderrs = [
    (np.std(q_scores) / np.sqrt(np.size(q_scores)))
    for quiz in University_Name_q_scores_per_quiz
    for q_scores in quiz
]

University_Name_condition_stats, University_Name_question_stats = grading_stats.score_stats(
    University_Name_avg_grade_per_q,
    University_Name_avg_grade_per_q_stds,
    University_Name_avg_grade_per_q_stderrs,
    experiment_conditions,
    questions_per_quiz,
)

incorrect_response_categories = {
    "Claude 3 Opus": [0, 0, 0, 0],
    "GPT-4": [0, 0, 0, 0],
    "Human": [0, 0, 0, 0],
}  # "*", "C K E", "Q Z I", "c c"

def classify_incorrect_responses(incorrect_responses, model_name):
    for i in range(len(incorrect_responses)):
        if "*" in incorrect_responses[i]:
            incorrect_response_categories[model_name][0] += 1
        elif "K" in incorrect_responses[i]:
            incorrect_response_categories[model_name][1] += 1
        elif "Z" in incorrect_responses[i]:
            incorrect_response_categories[model_name][2] += 1
        elif (incorrect_responses[i].count("c") >= 2) or (
            incorrect_responses[i].count("C") >= 2
        ):
            incorrect_response_categories[model_name][3] += 1
    return None


def classify_model_incorrect_responses(incorrect_responses, model_name):
    for i in range(len(incorrect_responses)):
        for _, _, answer, _ in incorrect_responses[i]:
            if "*" in answer:
                incorrect_response_categories[model_name][0] += 1
            elif "K" in answer:
                incorrect_response_categories[model_name][1] += 1
            elif "Z" in answer:
                incorrect_response_categories[model_name][2] += 1
            elif (answer.count("c") >= 2) or (answer.count("C") >= 2):
                incorrect_response_categories[model_name][3] += 1
    return None


classify_model_incorrect_responses(gpt4_arrows_incorrect_responses, "GPT-4")
classify_model_incorrect_responses(claude3opus_incorrect_responses, "Claude 3 Opus")

incorrect_responses_np = np.asarray(University_Name_incorrect_responses)

classify_incorrect_responses(incorrect_responses_np[:, 1], "Human")

plotting.plot_wrong_answers_bar(incorrect_response_categories)

plotting.comparison_bar_plot(
    experiment_conditions,
    [
        University_Name_condition_stats["avg_condition_accuracy"],
        gpt4_arrows_condition_stats["avg_condition_accuracy"],
        gpt3_arrows_condition_stats["avg_condition_accuracy"],
        claude3opus_condition_stats["avg_condition_accuracy"],
        claude2_condition_stats["avg_condition_accuracy"],
        falcon40b_arrows_condition_stats["avg_condition_accuracy"],
        pythia12b_deduped_condition_stats["avg_condition_accuracy"],
    ],
    [
        University_Name_condition_stats["stderrs_per_condition"],
        gpt4_arrows_condition_stats["stderrs_per_condition"],
        gpt3_arrows_condition_stats["stderrs_per_condition"],
        claude3opus_condition_stats["stderrs_per_condition"],
        claude2_condition_stats["stderrs_per_condition"],
        falcon40b_arrows_condition_stats["stderrs_per_condition"],
        pythia12b_deduped_condition_stats["stderrs_per_condition"],
    ],
    [
        "Human",
        "GPT-4",
        "GPT-3",
        "Claude 3 Opus",
        "Claude 2",
        "Falcon-40B",
        "Pythia-12B-Deduped",
    ],
    "plots/phase_1/Aggregate_Accuracy_Comparison.png",
    y_lim=(0, 1),
    label_x=False,
)

myorder = [0, 3]
experiment_conditions_reordered = [experiment_conditions[i] for i in myorder]

University_Name_avgs_reordered = [
    University_Name_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

University_Name_stderrs_reordered = [
    University_Name_condition_stats["stderrs_per_condition"][i] for i in myorder
]

University_Name_ind_reordered = [University_Name_individual_scores[i] for i in myorder]

plotting.bar_plot(
    experiment_conditions_reordered,
    University_Name_avgs_reordered,
    University_Name_stderrs_reordered,
    "plots/phase_1/Human_Results_defaults.png",
    fig_dimensions=[14, 8],
    title="Human Scores",
    ind_data=University_Name_ind_reordered,
)

raincloud_df = pd.DataFrame(
    [
        ("", ["Defaults", "Permuted Questions"][i], score)
        for i in range(2)
        for score in University_Name_ind_reordered[i]
    ],
    columns=["", "Baseline Condition", "Quiz Accuracy"],
)

myorder = [2, 1, 6, 5, 4, 7]
experiment_conditions_reordered = [experiment_conditions[i] for i in myorder]

University_Name_avgs_reordered = [
    University_Name_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

University_Name_stderrs_reordered = [
    University_Name_condition_stats["stderrs_per_condition"][i] for i in myorder
]

University_Name_ind_reordered = [University_Name_individual_scores[i] for i in myorder]

plotting.bar_plot(
    experiment_conditions_reordered,
    University_Name_avgs_reordered,
    University_Name_stderrs_reordered,
    "plots/phase_1/Human_Results_controls.png",
    fig_dimensions=[14, 8],
    title="Human Scores",
    ind_data=University_Name_ind_reordered,
)

plotting.comparison_bar_plot(
    experiment_conditions,
    [
        [np.mean(condition) if condition else 0 for condition in quiz]
        for quiz in University_Name_scores_grouped_by_quiz
    ],
    [
        [
            (np.std(condition) / np.sqrt(np.size(condition))) if condition else 0
            for condition in quiz
        ]
        for quiz in University_Name_scores_grouped_by_quiz
    ],
    [
        "Quiz 1",
        "Quiz 2",
        "Quiz 3",
        "Quiz 4",
    ],
    "plots/phase_1/Human_Accuracy_by_Quiz.png",
    y_lim=(0, 1),
    ind_data=University_Name_scores_grouped_by_quiz,
)

plotting.comparison_bar_plot(
    experiment_conditions,
    [
        prolific_condition_stats["avg_condition_accuracy"],
        University_Name_condition_stats["avg_condition_accuracy"],
    ],
    [
        prolific_condition_stats["stderrs_per_condition"],
        University_Name_condition_stats["stderrs_per_condition"],
    ],
    ["Prolific", "University students"],
    "plots/phase_1/Prolific_University_Name_Comparison.png",
    ind_data=[prolific_individual_scores, University_Name_individual_scores],
    y_lim=(0, 1),
    label_x=False,
    color="red",
)

myorder = [0]

experiment_conditions_reordered = [experiment_conditions[i] for i in myorder]

pythia_avgs_reordered = [
    pythia12b_deduped_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

pythia_stderrs_reordered = [
    pythia12b_deduped_condition_stats["stderrs_per_condition"][i] for i in myorder
]

falcon_avgs_reordered = [
    falcon40b_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

falcon_stderrs_reordered = [
    falcon40b_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

gpt3_avgs_reordered = [
    gpt3_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

gpt3_stderrs_reordered = [
    gpt3_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

gpt4_avgs_reordered = [
    gpt4_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

gpt4_stderrs_reordered = [
    gpt4_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

claude2_avgs_reordered = [
    claude2_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

claude2_stderrs_reordered = [
    claude2_condition_stats["stderrs_per_condition"][i] for i in myorder
]

claude3opus_avgs_reordered = [
    claude3opus_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

claude3opus_stderrs_reordered = [
    claude3opus_condition_stats["stderrs_per_condition"][i] for i in myorder
]

University_Name_avgs_reordered = [
    University_Name_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

University_Name_stderrs_reordered = [
    University_Name_condition_stats["stderrs_per_condition"][i] for i in myorder
]


# plot only performance in defaults
plotting.comparison_bar_plot_with_second_axis(
    [
        pythia_avgs_reordered,
        falcon_avgs_reordered,
        gpt3_avgs_reordered,
        claude2_avgs_reordered,
        gpt4_avgs_reordered,
        claude3opus_avgs_reordered,
        University_Name_avgs_reordered,
    ],
    [
        pythia_stderrs_reordered,
        falcon_stderrs_reordered,
        gpt3_stderrs_reordered,
        claude2_stderrs_reordered,
        gpt4_stderrs_reordered,
        claude3opus_stderrs_reordered,
        University_Name_stderrs_reordered,
    ],
    [
        "Pythia-12B-Deduped*",
        "Falcon-40B",
        "GPT-3",
        "Claude 2",
        "GPT-4",
        "Claude 3 Opus",
        "Human*",
    ],
    "plots/phase_1/Aggregate_Accuracy_Comparison_Default_MMLU.png",
    x_label="MMLU score",
    y_label="Accuracy",
    fig_size=[14, 12],
    title_size=24,
    axis_label_size=28,
    axis_tick_size=18,
)

myorder = [2, 1, 6, 5, 4, 7]

experiment_conditions_reordered = [experiment_conditions[i] for i in myorder]

pythia_avgs_reordered = [
    pythia12b_deduped_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

pythia_stderrs_reordered = [
    pythia12b_deduped_condition_stats["stderrs_per_condition"][i] for i in myorder
]

falcon_avgs_reordered = [
    falcon40b_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

falcon_stderrs_reordered = [
    falcon40b_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

gpt3_avgs_reordered = [
    gpt3_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

gpt3_stderrs_reordered = [
    gpt3_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

gpt4_avgs_reordered = [
    gpt4_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

gpt4_stderrs_reordered = [
    gpt4_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

claude2_avgs_reordered = [
    claude2_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

claude2_stderrs_reordered = [
    claude2_condition_stats["stderrs_per_condition"][i] for i in myorder
]

claude3opus_avgs_reordered = [
    claude3opus_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

claude3opus_stderrs_reordered = [
    claude3opus_condition_stats["stderrs_per_condition"][i] for i in myorder
]

University_Name_avgs_reordered = [
    University_Name_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

University_Name_stderrs_reordered = [
    University_Name_condition_stats["stderrs_per_condition"][i] for i in myorder
]

plotting.comparison_bar_plot(
    experiment_conditions_reordered,
    [
        University_Name_avgs_reordered,
        gpt4_avgs_reordered,
    ],
    [
        University_Name_stderrs_reordered,
        gpt4_stderrs_reordered,
    ],
    [
        "Human",
        "GPT-4",
    ],
    "plots/phase_1/Aggregate_Accuracy_Comparison_controls.png",
    y_lim=(0, 1),
)

# just default and permuted pairs, with outlines

myorder = [0, 2, 3, 1, 6, 5, 4, 7]

experiment_conditions_reordered = [experiment_conditions[i] for i in myorder]

gpt4_avgs_reordered = [
    gpt4_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

gpt4_stderrs_reordered = [
    gpt4_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

claude3opus_avgs_reordered = [
    claude3opus_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

claude3opus_stderrs_reordered = [
    claude3opus_condition_stats["stderrs_per_condition"][i] for i in myorder
]

University_Name_avgs_reordered = [
    University_Name_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

University_Name_stderrs_reordered = [
    University_Name_condition_stats["stderrs_per_condition"][i] for i in myorder
]

# Defaults and permuted pairs

x_values = np.arange(3)  # Generate evenly spaced values for x-axis
width = 0.7  # Width of each bar

# Set up the figure and axes
fig, ax = plt.subplots()

def_data = [
    University_Name_avgs_reordered[0],
    gpt4_avgs_reordered[0],
    claude3opus_avgs_reordered[0],
]

def_errors = [
    University_Name_stderrs_reordered[0],
    gpt4_stderrs_reordered[0],
    claude3opus_stderrs_reordered[0],
]

pp_data = [
    University_Name_avgs_reordered[1],
    gpt4_avgs_reordered[1],
    claude3opus_avgs_reordered[1],
]

pp_errors = [
    University_Name_stderrs_reordered[1],
    gpt4_stderrs_reordered[1],
    claude3opus_stderrs_reordered[1],
]

ct_bars = ax.bar(
    x_values,
    pp_data,
    width,
    yerr=pp_errors,
    capsize=2,
    label="Permuted pairs",
    color="#56C1FF",
)  # multi attribute
gt_bars = ax.bar(
    x_values,
    def_data,
    width,
    yerr=def_errors,
    capsize=2,
    label="Defaults",
    fill=False,
    edgecolor="black",
    linewidth=2,
)  # categorial
ax.set_ylim(0, 1.19)


# Add x-axis labels
ax.set_xticks(x_values)
ax.set_xticklabels(
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    rotation=0,
    fontsize=28,
)

ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=18)

plt.ylabel("Accuracy", fontsize=28)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    fontsize=28,
)


plt.savefig("plots/phase_1/fig3_new.png")

plt.clf()

# default, relational

x_values = np.arange(3)  # Generate evenly spaced values for x-axis
width = 0.7  # Width of each bar

# Set up the figure and axes
fig, ax = plt.subplots()

def_data = [
    University_Name_avgs_reordered[0],
    gpt4_avgs_reordered[0],
    claude3opus_avgs_reordered[0],
]

def_errors = [
    University_Name_stderrs_reordered[0],
    gpt4_stderrs_reordered[0],
    claude3opus_stderrs_reordered[0],
]

relational_data = [
    0.6538461538461539,
    0.325,
    0.7,
]  # printed out from phase 2, copied over
relational_errors = [
    0.09630166153112306,
    0.047434164902525694,
    0.07071067811865477,
]  # printed out from phase 2, copied over

ct_bars = ax.bar(
    x_values,
    relational_data,
    width,
    yerr=relational_errors,
    capsize=2,
    label="Relational",
    color="#56C1FF",
)  # multi attribute
gt_bars = ax.bar(
    x_values,
    def_data,
    width,
    yerr=def_errors,
    capsize=2,
    label="Defaults",
    fill=False,
    edgecolor="black",
    linewidth=2,
)  # categorial
ax.set_ylim(0, 1.19)


# Add x-axis labels
ax.set_xticks(x_values)
ax.set_xticklabels(
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    rotation=0,
    fontsize=28,
)

ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=18)

plt.ylabel("Accuracy", fontsize=28)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    fontsize=28,
)

plt.savefig("plots/phase_1/fig5_new.png")

plt.clf()


# onlyrhs, randoms, and randomfinals

x_values = np.arange(3)  # Generate evenly spaced values for x-axis
width = 0.7  # Width of each bar

# Set up the figure and axes
fig, ax = plt.subplots()

onlyrhs_data = [
    University_Name_avgs_reordered[4],
    gpt4_avgs_reordered[4],
    claude3opus_avgs_reordered[4],
]

onlyrhs_errors = [
    University_Name_stderrs_reordered[4],
    gpt4_stderrs_reordered[4],
    claude3opus_stderrs_reordered[4],
]

randoms_data = [
    University_Name_avgs_reordered[5],
    gpt4_avgs_reordered[5],
    claude3opus_avgs_reordered[5],
]

randoms_errors = [
    University_Name_stderrs_reordered[5],
    gpt4_stderrs_reordered[5],
    claude3opus_stderrs_reordered[5],
]

rf_data = [
    University_Name_avgs_reordered[7],
    gpt4_avgs_reordered[7],
    claude3opus_avgs_reordered[7],
]

rf_errors = [
    University_Name_stderrs_reordered[7],
    gpt4_stderrs_reordered[7],
    claude3opus_stderrs_reordered[7],
]

ct_bars = ax.bar(
    x_values,
    randoms_data,
    width,
    yerr=randoms_errors,
    capsize=2,
    label="Randoms",
    color="#ED908D",
)
ct_bars = ax.bar(
    x_values,
    rf_data,
    width,
    yerr=rf_errors,
    capsize=2,
    label="Random finals",
    color="#56C1FF",
)
gt_bars = ax.bar(
    x_values,
    onlyrhs_data,
    width,
    yerr=onlyrhs_errors,
    capsize=2,
    label="Only RHS",
    fill=False,
    edgecolor="black",
    linewidth=2,
)
ax.set_ylim(0, 1.19)


# Add x-axis labels
ax.set_xticks(x_values)
ax.set_xticklabels(
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    rotation=0,
    fontsize=28,
)

ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=18)

plt.ylabel("Accuracy", fontsize=28)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 0, 1]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    fontsize=28,
)

plt.savefig("plots/phase_1/fig4_new.png")

plt.clf()


myorder = [0, 3, 2, 1, 6, 5, 4, 7]

experiment_conditions_reordered = [experiment_conditions[i] for i in myorder]

gpt4_avgs_reordered = [
    gpt4_arrows_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

gpt4_stderrs_reordered = [
    gpt4_arrows_condition_stats["stderrs_per_condition"][i] for i in myorder
]

claude3opus_avgs_reordered = [
    claude3opus_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

claude3opus_stderrs_reordered = [
    claude3opus_condition_stats["stderrs_per_condition"][i] for i in myorder
]

University_Name_avgs_reordered = [
    University_Name_condition_stats["avg_condition_accuracy"][i] for i in myorder
]

University_Name_stderrs_reordered = [
    University_Name_condition_stats["stderrs_per_condition"][i] for i in myorder
]

# create side-by-side bar chart aggregate accuracy
plotting.comparison_bar_plot(
    experiment_conditions_reordered,
    [
        University_Name_avgs_reordered,
        gpt4_avgs_reordered,
        claude3opus_avgs_reordered,
    ],
    [
        University_Name_stderrs_reordered,
        gpt4_stderrs_reordered,
        claude3opus_stderrs_reordered,
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3 Opus",
    ],
    "plots/phase_1/Top_3_Comparison.png",
    title="Accuracy by Condition (Experiment 1)",
    y_lim=(0, 1),
    title_size=20,
    axis_tick_size=16,
    axis_label_size=16,
    legend_size=16,
)

# get the average accuracy (averaged over the average accuracy for each condition)
University_Name_avg_avg = np.mean(University_Name_avgs_reordered)
gpt4_avg_avg = np.mean(gpt4_avgs_reordered)

# get differences from the average
University_Name_diffs = [x - University_Name_avg_avg for x in University_Name_avgs_reordered]
gpt4_diffs = [x - gpt4_avg_avg for x in gpt4_avgs_reordered]

# get the average absolute difference from the average
University_Name_avg_abs_diff = np.mean([abs(x) for x in University_Name_diffs])
gpt4_avg_abs_diff = np.mean([abs(x) for x in gpt4_diffs])

University_Name_diffs.append(University_Name_avg_abs_diff)
gpt4_diffs.append(gpt4_avg_abs_diff)

University_Name_stderrs_reordered.append(0)
gpt4_stderrs_reordered.append(0)

# plot differences from self in defaults

University_Name_diffs_from_defaults = [
    x - University_Name_avgs_reordered[0] for x in University_Name_avgs_reordered
]

University_Name_diffs_from_defaults_stderrs = np.ndarray.tolist(
    np.sqrt(
        [
            (((x**2) + (University_Name_stderrs_reordered[0] ** 2)))
            for x in University_Name_stderrs_reordered[1:8]
        ]
    )
)

gpt4_diffs_from_defaults = [x - gpt4_avgs_reordered[0] for x in gpt4_avgs_reordered]

gpt4_diffs_from_defaults_stderrs = np.ndarray.tolist(
    np.sqrt(
        [
            (((x**2) + (gpt4_stderrs_reordered[0] ** 2)))
            for x in gpt4_stderrs_reordered[1:8]
        ]
    )
)

claude3opus_diffs_from_defaults = [
    x - claude3opus_avgs_reordered[0] for x in claude3opus_avgs_reordered
]

claude3opus_diffs_from_defaults_stderrs = np.ndarray.tolist(
    np.sqrt(
        [
            (((x**2) + (claude3opus_stderrs_reordered[0] ** 2)))
            for x in claude3opus_stderrs_reordered[1:8]
        ]
    )
)


plotting.comparison_bar_plot(
    experiment_conditions_reordered[1:8],
    [
        University_Name_diffs_from_defaults[1:8],
        gpt4_diffs_from_defaults[1:8],
        claude3opus_diffs_from_defaults[1:8],
    ],
    [
        University_Name_diffs_from_defaults_stderrs,
        gpt4_diffs_from_defaults_stderrs,
        claude3opus_diffs_from_defaults_stderrs,
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_1/Performance_Differences_phase1.png",
    title="",
    label_x=False,
    y_lim=(-1, 0.25),
    fig_size=[14, 12],
    title_size=24,
    axis_label_size=18,
    axis_tick_size=18,
    legend_size=16,
)

plotting.comparison_bar_plot(
    experiment_conditions_reordered[1:8],
    [
        University_Name_diffs_from_defaults[1:8],
        gpt4_diffs_from_defaults[1:8],
        claude3opus_diffs_from_defaults[1:8],
    ],
    [
        University_Name_diffs_from_defaults_stderrs,
        gpt4_diffs_from_defaults_stderrs,
        claude3opus_diffs_from_defaults_stderrs,
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_1/Performance_Differences_by_Participant_phase1.png",
    title="Accuracy Relative to Defaults",
    transpose=True,
    label_x=False,
    y_lim=(-1, 0.25),
    title_size=24,
    axis_label_size=18,
    axis_tick_size=18,
    legend_size=16,
)

plotting.comparison_bar_plot(
    experiment_conditions_reordered[1:4],
    [
        University_Name_diffs_from_defaults[1:4],
        gpt4_diffs_from_defaults[1:4],
        claude3opus_diffs_from_defaults[1:4],
    ],
    [
        University_Name_diffs_from_defaults_stderrs[0:3],
        gpt4_diffs_from_defaults_stderrs[0:3],
        claude3opus_diffs_from_defaults_stderrs[0:3],
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_1/Performance_Differences_phase1_first_half.png",
    title="",
    label_x=False,
    y_lim=(-0.5, 0.25),
    title_size=24,
    axis_label_size=18,
    axis_tick_size=18,
    legend_size=16,
)

plotting.comparison_bar_plot(
    experiment_conditions_reordered[1:4],
    [
        University_Name_diffs_from_defaults[1:4],
        gpt4_diffs_from_defaults[1:4],
        claude3opus_diffs_from_defaults[1:4],
    ],
    [
        University_Name_diffs_from_defaults_stderrs[0:3],
        gpt4_diffs_from_defaults_stderrs[0:3],
        claude3opus_diffs_from_defaults_stderrs[0:3],
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_1/Performance_Differences_by_Participant_phase1_first_half.png",
    title="Accuracy Relative to Defaults",
    transpose=True,
    label_x=False,
    y_lim=(-0.5, 0.25),
    title_size=24,
    axis_label_size=18,
    axis_tick_size=18,
    legend_size=16,
)

plotting.comparison_bar_plot(
    experiment_conditions_reordered[4:8],
    [
        University_Name_diffs_from_defaults[4:8],
        gpt4_diffs_from_defaults[4:8],
        claude3opus_diffs_from_defaults[4:8],
    ],
    [
        University_Name_diffs_from_defaults_stderrs[3:7],
        gpt4_diffs_from_defaults_stderrs[3:7],
        claude3opus_diffs_from_defaults_stderrs[3:7],
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_1/Performance_Differences_phase1_onlyrhs_randoms.png",
    title="",
    label_x=False,
    y_lim=(-1, 0.25),
    title_size=24,
    axis_label_size=18,
    axis_tick_size=18,
    legend_size=16,
)

plotting.comparison_bar_plot(
    experiment_conditions_reordered[4:8],
    [
        University_Name_diffs_from_defaults[4:8],
        gpt4_diffs_from_defaults[4:8],
        claude3opus_diffs_from_defaults[4:8],
    ],
    [
        University_Name_diffs_from_defaults_stderrs[3:7],
        gpt4_diffs_from_defaults_stderrs[3:7],
        claude3opus_diffs_from_defaults_stderrs[3:7],
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_1/Performance_Differences_by_Participant_phase1_onlyrhs_randoms.png",
    title="Accuracy Relative to Defaults",
    transpose=True,
    label_x=False,
    y_lim=(-1, 0.25),
    title_size=24,
    axis_label_size=18,
    axis_tick_size=18,
    legend_size=14,
)

plotting.comparison_bar_plot(
    experiment_conditions_reordered + ["average_abs Diff_from Average"],
    [
        University_Name_diffs,
        gpt4_diffs,
    ],
    [
        University_Name_stderrs_reordered,
        gpt4_stderrs_reordered,
    ],
    [
        "Human",
        "GPT-4",
    ],
    "plots/phase_1/Performance_Differences_from_Averages_phase1.png",
    y_lim=(-0.5, 0.5),
    fig_size=[14, 10],
    title_size=24,
    axis_label_size=16,
    axis_tick_size=16,
    legend_size=16,
)

plotting.comparison_bar_plot(
    [x.name for x in list(grading_stats.FailureMode)],
    [
        University_Name_failure_modes,
        gpt4_arrows_failure_modes,
    ],
    [0] * 4,
    ["Human", "GPT-4"],
    "plots/phase_1/Failure_Modes.png",
    title="Failure Modes",
    y_label="Percent of Incorrect Responses",
)

# # Code below is used to generate Failure Modes Table
# # Commented out to speed up the code
# fail_modes = (
#     pd.DataFrame(
#         {
#             "Mode": [
#                 " ".join(x.name.split("_")).title()
#                 for x in list(grading_stats.FailureMode)
#             ],
#             "Human": University_Name_failure_modes,
#             "GPT-4": gpt4_arrows_failure_modes,
#             "Claude 3 Opus": claude3opus_failure_modes,
#         }
#     )
#     .set_index("Mode")
#     .T.round(3)
# )
# print(fail_modes.to_latex())

plotting.bar_plot(
    [x.name for x in list(grading_stats.FailureMode)],
    University_Name_failure_modes,
    [0] * 4,
    "plots/phase_1/Human_Failure_Modes.png",
    title="Human Failure Modes",
    y_label="Percent of Incorrect Responses",
)


# plot distribution of human performance

plotting.human_accuracy_raincloud_plot(
    University_Name_df, experiment_conditions, "plots/phase_1/human_acc_raincloud.png"
)


# plot accuracy versus time

plotting.human_accuracy_v_duration_plot(
    University_Name_df, "plots/phase_1/human_accuracy_v_duration.png"
)

# plot line of best fit graphs


human_q1_vals = []
gpt4_q1_vals = []
claude3opus_q1_vals = []

human_q1_errs = []
gpt4_q1_errs = []
claude3opus_q1_errs = []

human_q4_vals = []
gpt4_q4_vals = []
claude3opus_q4_vals = []

human_q4_errs = []
gpt4_q4_errs = []
claude3opus_q4_errs = []

counter = 1


def subplots_best_fit_and_points_plot(
    slope: Number,
    intercept: Number,
    r: Number,
    y_vals: List[Number],
    y_err: list,
    i: Number,
    title: str,
    suppress_xticks: bool,
    counter: int,
    compile_first_last_data: bool,
    nrows: int = 2,
    ncols: int = 8,
):
    x = np.array([0, len(y_vals) - 1])
    y = slope * x + intercept
    plt.subplot(nrows, ncols, i)
    if title:
        plt.title(title, fontdict={"fontsize": 20})
    plt.plot(
        x,
        y,
        "-b",
        label=f"y={round(slope, 2)}x+{round(intercept, 2)}, r={round(r, 2)}",
    )
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(-0.5, 3.5)

    labels = ["Q1", "Q2", "Q3", "Q4"] if not suppress_xticks else [""] * 4
    plt.xticks(ticks=[0, 1, 2, 3], labels=labels)

    if i % ncols != 1:
        plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[""] * 6)
    x = np.arange(len(y_vals))
    plt.scatter(x, y_vals)
    plt.errorbar(x, y_vals, yerr=y_err, fmt="o", elinewidth=4, capsize=7)
    if compile_first_last_data:
        if counter == 1:
            human_q1_vals.append(y_vals[0])
            human_q1_errs.append(y_err[0])
            human_q4_vals.append(y_vals[3])
            human_q4_errs.append(y_err[3])
        if counter == 2:
            claude3opus_q1_vals.append(y_vals[0])
            claude3opus_q1_errs.append(y_err[0])
            claude3opus_q4_vals.append(y_vals[3])
            claude3opus_q4_errs.append(y_err[3])
        if counter == 3:
            gpt4_q1_vals.append(y_vals[0])
            gpt4_q1_errs.append(y_err[0])
            gpt4_q4_vals.append(y_vals[3])
            gpt4_q4_errs.append(y_err[3])

        counter += 1
        if counter == 4:
            counter = 1
    return counter


plt.clf()

fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(34, 27))

fig.suptitle(
    "Accuracy vs. Question Number by Subject Type and Experiment Condition",
    fontsize=27,
)

for ax, row in zip(
    axes[:, 0], ["Human".ljust(20), "Claude 3 Opus".ljust(30), "GPT-4".ljust(20)]
):
    ax.set_ylabel(row, rotation=0, fontsize=20)

for list_of_axes in axes:
    for ax in list_of_axes:
        ax.tick_params(axis="both", which="major", labelsize=16)

for i in range(len(experiment_conditions)):
    for j, stats in enumerate(
        [
            University_Name_question_stats,
            claude3opus_question_stats,
            gpt4_arrows_question_stats,
        ]
    ):
        counter = subplots_best_fit_and_points_plot(
            slope=stats["regressions"][i].slope,
            intercept=stats["regressions"][i].intercept,
            r=stats["regressions"][i].rvalue,
            y_vals=stats["avg_q_accuracy"][i],
            y_err=stats["stderrs_per_q_num"][i],
            i=(i + 1 + (8 * j)),
            title=experiment_conditions[i] if j == 0 else "",
            suppress_xticks=False,
            counter=counter,
            compile_first_last_data=True,
            nrows=3,
            ncols=8,
        )

human_q1_vals = [
    human_q1_vals[0],
    human_q1_vals[3],
    human_q1_vals[2],
    human_q1_vals[1],
    human_q1_vals[6],
    human_q1_vals[5],
    human_q1_vals[4],
    human_q1_vals[7],
]

human_q1_errs = [
    human_q1_errs[0],
    human_q1_errs[3],
    human_q1_errs[2],
    human_q1_errs[1],
    human_q1_errs[6],
    human_q1_errs[5],
    human_q1_errs[4],
    human_q1_errs[7],
]

human_q4_vals = [
    human_q4_vals[0],
    human_q4_vals[3],
    human_q4_vals[2],
    human_q4_vals[1],
    human_q4_vals[6],
    human_q4_vals[5],
    human_q4_vals[4],
    human_q4_vals[7],
]

human_q4_errs = [
    human_q4_errs[0],
    human_q4_errs[3],
    human_q4_errs[2],
    human_q4_errs[1],
    human_q4_errs[6],
    human_q4_errs[5],
    human_q4_errs[4],
    human_q4_errs[7],
]


gpt4_q1_vals = [
    gpt4_q1_vals[0],
    gpt4_q1_vals[3],
    gpt4_q1_vals[2],
    gpt4_q1_vals[1],
    gpt4_q1_vals[6],
    gpt4_q1_vals[5],
    gpt4_q1_vals[4],
    gpt4_q1_vals[7],
]

gpt4_q1_errs = [
    gpt4_q1_errs[0],
    gpt4_q1_errs[3],
    gpt4_q1_errs[2],
    gpt4_q1_errs[1],
    gpt4_q1_errs[6],
    gpt4_q1_errs[5],
    gpt4_q1_errs[4],
    gpt4_q1_errs[7],
]

gpt4_q4_vals = [
    gpt4_q4_vals[0],
    gpt4_q4_vals[3],
    gpt4_q4_vals[2],
    gpt4_q4_vals[1],
    gpt4_q4_vals[6],
    gpt4_q4_vals[5],
    gpt4_q4_vals[4],
    gpt4_q4_vals[7],
]

gpt4_q4_errs = [
    gpt4_q4_errs[0],
    gpt4_q4_errs[3],
    gpt4_q4_errs[2],
    gpt4_q4_errs[1],
    gpt4_q4_errs[6],
    gpt4_q4_errs[5],
    gpt4_q4_errs[4],
    gpt4_q4_errs[7],
]


claude3opus_q1_vals = [
    claude3opus_q1_vals[0],
    claude3opus_q1_vals[3],
    claude3opus_q1_vals[2],
    claude3opus_q1_vals[1],
    claude3opus_q1_vals[6],
    claude3opus_q1_vals[5],
    claude3opus_q1_vals[4],
    claude3opus_q1_vals[7],
]

claude3opus_q1_errs = [
    claude3opus_q1_errs[0],
    claude3opus_q1_errs[3],
    claude3opus_q1_errs[2],
    claude3opus_q1_errs[1],
    claude3opus_q1_errs[6],
    claude3opus_q1_errs[5],
    claude3opus_q1_errs[4],
    claude3opus_q1_errs[7],
]

claude3opus_q4_vals = [
    claude3opus_q4_vals[0],
    claude3opus_q4_vals[3],
    claude3opus_q4_vals[2],
    claude3opus_q4_vals[1],
    claude3opus_q4_vals[6],
    claude3opus_q4_vals[5],
    claude3opus_q4_vals[4],
    claude3opus_q4_vals[7],
]

claude3opus_q4_errs = [
    claude3opus_q4_errs[0],
    claude3opus_q4_errs[3],
    claude3opus_q4_errs[2],
    claude3opus_q4_errs[1],
    claude3opus_q4_errs[6],
    claude3opus_q4_errs[5],
    claude3opus_q4_errs[4],
    claude3opus_q4_errs[7],
]


q1_vals = [human_q1_vals, gpt4_q1_vals, claude3opus_q1_vals]

q1_errs = [human_q1_errs, gpt4_q1_errs, claude3opus_q1_errs]

q4_vals = [human_q4_vals, gpt4_q4_vals, claude3opus_q4_vals]

q4_errs = [human_q4_errs, gpt4_q4_errs, claude3opus_q4_errs]

plt.tight_layout(pad=3)

plt.savefig("plots/phase_1/top_performers_best_fit_and_points.png")
plt.close()

# plot overall performance for top three, only q1s then only q4s

plotting.comparison_bar_plot(
    experiment_conditions_reordered,
    q1_vals,
    q1_errs,
    [
        "Human",
        "GPT-4",
        "Claude 3 Opus",
    ],
    "plots/phase_1/Top_3_Comparison_q1s.png",
    y_lim=(0, 1),
)

plt.clf()

plotting.comparison_bar_plot(
    experiment_conditions_reordered,
    q4_vals,
    q4_errs,
    [
        "Human",
        "GPT-4",
        "Claude 3 Opus",
    ],
    "plots/phase_1/Top_3_Comparison_q4s.png",
    y_lim=(0, 1),
)


fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(22, 6))

fig.suptitle(
    "Human Accuracy versus Question Number by Experiment Condition",
    fontsize=27,
)

for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=16)

axes[0].set_ylabel("Human        ", rotation=0, fontsize=20)

for i in range(len(experiment_conditions)):
    counter = subplots_best_fit_and_points_plot(
        slope=University_Name_question_stats["regressions"][i].slope,
        intercept=University_Name_question_stats["regressions"][i].intercept,
        r=University_Name_question_stats["regressions"][i].rvalue,
        y_vals=University_Name_question_stats["avg_q_accuracy"][i],
        y_err=University_Name_question_stats["stderrs_per_q_num"][i],
        i=(i + 1),
        title="",
        counter=counter,
        compile_first_last_data=False,
        suppress_xticks=False,
        nrows=1,
    )

plt.savefig("plots/phase_1/Human_ICL_by_Condition.png")
plt.close()


################################
# STATISTICAL WORK STARTS HERE #
################################

University_Name_df["respondent_scores"] = University_Name_df.apply(
    lambda row: score_respondent_all(
        row,
        University_Name_q_scores_per_quiz,
        University_Name_incorrect_responses,
        University_Name_failure_modes,
    ),
    axis=1,
)


prolific_df["respondent_scores"] = prolific_df.apply(
    lambda row: score_respondent_all(
        row,
        prolific_q_scores_per_quiz,
        prolific_incorrect_responses,
        prolific_failure_modes,
    ),
    axis=1,
)

prolific_df = prolific_df.dropna(subset=["respondent_score"])

prolific_df_exploded = prolific_df.explode("respondent_scores")

prolific_df_exploded["subject_type"] = pd.Series(
    ["Prolific" for x in range(len(prolific_df_exploded.index))]
)

University_Name_df = University_Name_df.dropna(subset=["respondent_score"])

University_Name_df_exploded = University_Name_df.explode("respondent_scores")

University_Name_df_exploded["subject_type"] = pd.Series(
    ["human" for x in range(len(University_Name_df_exploded.index))]
)

human_df_exploded = pd.concat([prolific_df_exploded, University_Name_df_exploded])

human_df_exploded["respondent_scores"] = pd.to_numeric(
    human_df_exploded["respondent_scores"]
)

q_nums = cycle([1, 2, 3, 4])
human_df_exploded["question_num"] = [
    next(q_nums) for num in range(len(human_df_exploded))
]

human_df_subset = human_df_exploded[
    ["subject_type", "quiz_number", "quiz_class", "respondent_scores", "question_num"]
]

all_subjects_df = pd.concat([all_subjects_df, human_df_subset])

all_subjects_df = all_subjects_df.dropna(
    subset=["quiz_number"]
)

all_subjects_df["quiz_number"] = all_subjects_df["quiz_number"].astype(int)

all_subjects_df["respondent_scores"] = all_subjects_df["respondent_scores"].astype(int)

all_subjects_df["question_num"] = all_subjects_df["question_num"].astype(int)

all_subjects_df = all_subjects_df[all_subjects_df.subject_type != "Pythia-410M-Deduped"]

all_subjects_df = all_subjects_df[all_subjects_df.subject_type != "Pythia-70M-Deduped"]

all_subjects_df["quiz_number_modded"] = all_subjects_df["quiz_number"].apply(
    quiz_to_quiz_modded
)


# Reset dataframe index

# Clean up the index, which retains history of the concatenations
all_subjects_df.reset_index(inplace=True, drop=True)


# Stop pandas from truncating displays
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 1000)

model_name_stats = "GPT-4"  # "Claude-3-Opus"

all_subjects_df = all_subjects_df[
    (all_subjects_df.subject_type == model_name_stats)
    | (all_subjects_df.subject_type == "human")
]


human_idx = str(all_subjects_df["subject_type"].unique().tolist().index("human"))

# %% Begin modeling

print("Non-interacted model:")

permuted_qs_idx = str(4)

res_subjplusclass = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))+ C(quiz_class, Treatment(reference="
    + permuted_qs_idx
    + "))",
    data=all_subjects_df,
).fit(maxiter=1000, method="bfgs")

print(res_subjplusclass.summary())

print("Interacted model:")

res_subjXclass = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))*C(quiz_class, Treatment(reference="
    + permuted_qs_idx
    + "))",
    data=all_subjects_df,
).fit(maxiter=1000, method="bfgs")

print(res_subjXclass.summary())

degfree = 7

print("Degrees of freedom is " + str(degfree))

lik_ratio = degfree * (res_subjXclass.llf - res_subjplusclass.llf)

print("Likelihood ratio is " + str(lik_ratio))

chi_sq_p_value = chi2.sf(lik_ratio, degfree)

print(
    "p value for the significance of model improvement when including interaction terms is "
    + str(chi_sq_p_value)
)

print("Investigating simple effects")

all_conditions_df = all_subjects_df

quiz_class_inclusion_list = ["defaults", "permuted_questions"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

quiz_class_inclusion_list = ["defaults"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

quiz_class_inclusion_list = ["permuted_questions"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

quiz_class_inclusion_list = ["distracted"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())


quiz_class_inclusion_list = ["permuted_pairs"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())


quiz_class_inclusion_list = ["only_rhs"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

quiz_class_inclusion_list = ["random_permuted_pairs", "randoms", "random_finals"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

quiz_class_inclusion_list = [
    "randoms",
    "random_permuted_pairs",
]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

quiz_class_inclusion_list = ["random_finals"]

specific_conditions_df = all_subjects_df[
    all_subjects_df.quiz_class.isin(quiz_class_inclusion_list)
]

print(
    "Effect of subject with only the conditions " + str(quiz_class_inclusion_list) + ":"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(subject_type, Treatment(reference="
    + human_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())
