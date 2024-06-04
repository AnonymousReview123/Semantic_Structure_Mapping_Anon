import math
from itertools import cycle
from numbers import Number
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.distributions import chi2
from statsmodels.formula.api import ols

import grading_stats
import plotting

sns.set_palette("colorblind")

experiment_conditions = [
    "categorial",
    "multi_attribute",
    "numeric",
    "numeric_multi_attribute",
    "relational",
]


def quiz_to_condition(n: Number) -> str:
    n = int(n)
    return experiment_conditions[n // 2]


def overall_number_to_quiz_number(n: Number) -> int:
    """
    Takes a quiz number and returns the quiz number from 1 to 2.
    """
    n = int(n)
    return (n % 2) + 1


def quiz_to_quiz_modded(n: Number) -> Number:
    n = int(n)
    return (n % 2) + 1


def classify_quiz(row: pd.Series):
    quiz_num = row["quiz_number"]
    if math.isnan(quiz_num):
        return "NA"
    return quiz_to_condition(quiz_num)


num_quizzes = 10
questions_per_quiz = 4
samples_per_q = 5

# answer key as a flat list
answer_key: List[Tuple[str, bool]] = [
    answer for condition in grading_stats.PHASE_2_ANSWERS for answer in condition
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
        "quiz_results/phase_2/GPT/GPT-3/gpt-3_results_phase2.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="GPT-3",
    all_subjects_df=all_subjects_df,
    suppress_get_failure_modes=True,
)


(
    gpt4_arrows_condition_stats,
    gpt4_arrows_question_stats,
    gpt4_arrows_incorrect_responses,
    gpt4_arrows_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_2/GPT/GPT-4/gpt-4_results_phase2.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="GPT-4",
    all_subjects_df=all_subjects_df,
    suppress_get_failure_modes=True,
)

(
    falcon40b_arrows_condition_stats,
    falcon40b_arrows_question_stats,
    falcon40b_arrows_responses,
    falcon40b_arrows_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_2/Falcon/falcon_40b_results_phase2.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Falcon-40B-arrows",
    all_subjects_df=all_subjects_df,
    suppress_get_failure_modes=True,
)

(
    claude2_condition_stats,
    claude2_question_stats,
    claude2_incorrect_responses,
    claude2_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    ["quiz_results/phase_2/Claude/Claude 2/claude-2_results.txt"],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Claude-2",
    all_subjects_df=all_subjects_df,
    suppress_get_failure_modes=True,
    first_and_last=True,
)

(
    claude3opus_condition_stats,
    claude3opus_question_stats,
    claude3opus_incorrect_responses,
    claude3opus_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    ["quiz_results/phase_2/Claude/Claude 3 Opus/claude-3-opus-20240229_results.txt"],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Claude-3",
    all_subjects_df=all_subjects_df,
    suppress_get_failure_modes=True,
    first_and_last=True,
)

(
    pythia12b_deduped_condition_stats,
    pythia12b_deduped_question_stats,
    pythia12b_deduped_incorrect_responses,
    pythia12b_deduped_failure_modes,
    all_subjects_df,
) = grading_stats.model_stats(
    [
        "quiz_results/phase_2/Pythia/12b-deduped_results.txt",
    ],
    answer_key,
    experiment_conditions,
    num_quizzes,
    questions_per_quiz,
    samples_per_q,
    model_name="Pythia-12B-Deduped",
    all_subjects_df=all_subjects_df,
    suppress_get_failure_modes=True,
)

all_subjects_df["quiz_class"] = all_subjects_df.apply(classify_quiz, axis=1)


human_df = pd.read_csv(
    "quiz_results/phase_2/Human/Phase 2 (Students)_December 5, 2023_14.54.csv"
)

human_df["RecipientEmail"] = human_df["Q111"]

human_df = human_df.drop(columns=["Q111"])

human_df["RecipientEmail"] = human_df["RecipientEmail"].replace(np.nan, "")

human_df = human_df[human_df["RecipientEmail"].str.contains("@University_Name.edu")]

human_df = human_df.drop_duplicates(subset=["RecipientEmail"], keep="last")

human_df = human_df.rename(columns={"Q112": "Consent", "Q113": "Attention"})

def extract_quiz_number(row: pd.Series) -> int:
    if row["Finished"].lower() != "true":
        return None
    for i in range(10):
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

print("Num human subjects phase 2 with relational:")

print(human_df.shape[0])

print("Num human subjects phase 2 without relational:")

print(human_df[human_df.quiz_class != "relational"].shape[0])

print("Grouped by condition:")

print(human_df.groupby(['quiz_class']).size())

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
    incorrect_responses: List[Tuple[str, str, str]],
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
            if quiz_to_condition(quiz_number) == "multi_attribute":
                with open(
                    "data_printouts/phase_2/human_multi-attribute_errors.txt", "a"
                ) as f:
                    f.write(f"{questions[i]}  {response.ljust(20)}{answer[0]}\n")
    this_respondent_scores_arr = np.asarray(this_respondent_scores)
    return this_respondent_scores_arr

human_q_scores_per_quiz = [[[] for _ in range(4)] for _ in range(10)]
human_incorrect_responses: List[Tuple[str, str, str]] = []
human_failure_modes: List[int] = [0 for _ in range(len(grading_stats.FailureMode))]

human_individual_scores = [[] for _ in range(len(experiment_conditions))]
human_scores_grouped_by_quiz = [
    [[] for _ in range(len(experiment_conditions))] for _ in range(2)
]

human_df["respondent_score"] = human_df.apply(
    lambda row: score_respondent(
        row,
        human_q_scores_per_quiz,
        human_incorrect_responses,
        human_individual_scores,
        human_failure_modes,
        human_scores_grouped_by_quiz,
    ),
    axis=1,
)

human_avg_grade_per_q = [
    np.average(q_scores) for quiz in human_q_scores_per_quiz for q_scores in quiz
]

human_avg_grade_per_q_stds = [
    np.std(q_scores) for quiz in human_q_scores_per_quiz for q_scores in quiz
]

human_avg_grade_per_q_stderrs = [
    (np.std(q_scores) / np.sqrt(np.size(q_scores)))
    for quiz in human_q_scores_per_quiz
    for q_scores in quiz
]

human_condition_stats, human_question_stats = grading_stats.score_stats(
    human_avg_grade_per_q,
    human_avg_grade_per_q_stds,
    human_avg_grade_per_q_stderrs,
    experiment_conditions,
    questions_per_quiz,
)

human_means, human_stds, human_stderrs = [], [], []
human_individual_scores_np = np.array(human_individual_scores, dtype=object)

for i in range(human_individual_scores_np.shape[0]):
    human_means.append(np.mean(human_individual_scores_np[i]))
    human_stds.append(np.std(human_individual_scores_np[i]))
    human_stderrs.append(
        np.std(human_individual_scores_np[i])
        / np.sqrt(np.size(human_individual_scores_np[i]))
    )

plotting.bar_plot(
    experiment_conditions,
    human_means,  # human_condition_stats["avg_condition_accuracy"],
    human_stderrs,  # human_condition_stats["stdevs_per_condition"],
    "plots/phase_2/Human_Results_phase_2.png",
    fig_dimensions=[14, 8],
    title="Human Scores",
    ind_data=human_individual_scores,
)

plotting.comparison_bar_plot(
    experiment_conditions,
    [
        [np.mean(condition) for condition in quiz]
        for quiz in human_scores_grouped_by_quiz
    ],
    [
        [(np.std(condition) / np.sqrt(np.size(condition))) for condition in quiz]
        for quiz in human_scores_grouped_by_quiz
    ],
    [
        "Quiz 1",
        "Quiz 2",
    ],
    "plots/phase_2/Human_Accuracy_by_Quiz_phase_2.png",
    y_lim=(0, 1),
    ind_data=human_scores_grouped_by_quiz,
)

# create side-by-side bar chart aggregate accuracy
plotting.comparison_bar_plot(
    experiment_conditions,
    [
        human_means,
        gpt4_arrows_condition_stats["avg_condition_accuracy"],
        gpt3_arrows_condition_stats["avg_condition_accuracy"],
        claude3opus_condition_stats["avg_condition_accuracy"],
        claude2_condition_stats["avg_condition_accuracy"],
    ],
    [
        human_stderrs,
        gpt4_arrows_condition_stats["stderrs_per_condition"],
        gpt3_arrows_condition_stats["stdevs_per_condition"],
        claude3opus_condition_stats["stdevs_per_condition"],
        claude2_condition_stats["stdevs_per_condition"],
    ],
    [
        "Human",
        "GPT-4",
        "GPT-3",
        "Claude 3 Opus",
        "Claude 2",
    ],
    "plots/phase_2/Aggregate_Accuracy_Comparison.png",
    y_lim=(0, 1),
    label_x=False,
)


x_values = np.arange(3)  # Generate evenly spaced values for x-axis
width = 0.35  # Width of each bar

# Set up the figure and axes
fig, ax = plt.subplots()

cat_data = [
    human_means[0],
    gpt4_arrows_condition_stats["avg_condition_accuracy"][0],
    claude3opus_condition_stats["avg_condition_accuracy"][0],
]

cat_errors = [
    human_stderrs[0],
    gpt4_arrows_condition_stats["stderrs_per_condition"][0],
    claude3opus_condition_stats["stderrs_per_condition"][0],
]

ma_data = [
    human_means[1],
    gpt4_arrows_condition_stats["avg_condition_accuracy"][1],
    claude3opus_condition_stats["avg_condition_accuracy"][1],
]

ma_errors = [
    human_stderrs[1],
    gpt4_arrows_condition_stats["stderrs_per_condition"][1],
    claude3opus_condition_stats["stderrs_per_condition"][1],
]

num_data = [
    human_means[2],
    gpt4_arrows_condition_stats["avg_condition_accuracy"][2],
    claude3opus_condition_stats["avg_condition_accuracy"][2],
]

num_errors = [
    human_stderrs[2],
    gpt4_arrows_condition_stats["stderrs_per_condition"][2],
    claude3opus_condition_stats["stderrs_per_condition"][2],
]

nma_data = [
    human_means[3],
    gpt4_arrows_condition_stats["avg_condition_accuracy"][3],
    claude3opus_condition_stats["avg_condition_accuracy"][3],
]

nma_errors = [
    human_stderrs[3],
    gpt4_arrows_condition_stats["stderrs_per_condition"][3],
    claude3opus_condition_stats["stderrs_per_condition"][3],
]
ct_bars = ax.bar(
    x_values - width / 2,
    ma_data,
    width,
    yerr=ma_errors,
    capsize=2,
    label="Categorial",
    color="#56C1FF",
)  # multi attribute
ct_bars = ax.bar(
    x_values + width / 2,
    nma_data,
    width,
    yerr=nma_errors,
    capsize=2,
    label="Numeric",
    color="#ED908D",
)  # numeric multi attribute
gt_bars = ax.bar(
    x_values + width / 2,
    num_data,
    width,
    yerr=num_errors,
    capsize=2,
    label="Single attribute",
    fill=False,
    edgecolor="black",
    linewidth=2,
)  # numeric
gt_bars = ax.bar(
    x_values - width / 2,
    cat_data,
    width,
    yerr=cat_errors,
    capsize=2,
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
order = [2, 1, 0]
plt.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    fontsize=28,
)

plt.savefig("plots/phase_2/exp2_new.png")

plt.clf()

# create side-by-side bar chart aggregate accuracy
plotting.comparison_bar_plot(
    experiment_conditions[:-1],
    [
        human_means[:-1], 
        gpt4_arrows_condition_stats["avg_condition_accuracy"][:-1],
        claude3opus_condition_stats["avg_condition_accuracy"][:-1],
    ],
    [
        human_stderrs[:-1],
        gpt4_arrows_condition_stats["stderrs_per_condition"][:-1],
        claude3opus_condition_stats["stdevs_per_condition"][:-1],
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3 Opus",
    ],
    "plots/phase_2/Top_3_Comparison.png",
    title="",
    y_lim=(0, 1),
    label_x=False,
    title_size=20,
    axis_tick_size=16,
    axis_label_size=16,
    legend_size=16,
)


print(
    "Relational data: "
    + str(
        [
            human_means[-1:],
            gpt4_arrows_condition_stats["avg_condition_accuracy"][-1:],
            claude3opus_condition_stats["avg_condition_accuracy"][-1:],
        ]
    )
)

print(
    "Relational errors: "
    + str(
        [
            human_stderrs[-1:],
            gpt4_arrows_condition_stats["stderrs_per_condition"][-1:],
            claude3opus_condition_stats["stdevs_per_condition"][-1:],
        ]
    )
)

plotting.comparison_bar_plot(
    experiment_conditions[-1:],
    [
        human_means[-1:],
        gpt4_arrows_condition_stats["avg_condition_accuracy"][-1:],
        claude3opus_condition_stats["avg_condition_accuracy"][-1:],
    ],
    [
        human_stderrs[-1:],
        gpt4_arrows_condition_stats["stderrs_per_condition"][-1:],
        claude3opus_condition_stats["stdevs_per_condition"][-1:],
    ],
    [
        "Human",
        "GPT-4",
        "Claude 3",
    ],
    "plots/phase_2/Relational_Only_Top_3_Comparison.png",
    fig_size=[4, 8],
    title="Relational Condition Accuracy",
    label_x=False,
    y_lim=(0, 1),
)

################################
# STATISTICAL WORK STARTS HERE #
################################

human_df["respondent_scores"] = human_df.apply(
    lambda row: score_respondent_all(
        row,
        human_q_scores_per_quiz,
        human_incorrect_responses,
        human_failure_modes,
    ),
    axis=1,
)

human_df = human_df.dropna(subset=["respondent_score"])

human_df_exploded = human_df.explode("respondent_scores")

human_df_exploded["subject_type"] = pd.Series(
    ["human" for x in range(len(human_df_exploded.index))]
)

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

all_subjects_df = all_subjects_df[
    (all_subjects_df.subject_type ==  "GPT-4")#"Claude-3") 
    | (all_subjects_df.subject_type == "human")
]

all_subjects_df_with_relational = all_subjects_df

all_subjects_df = all_subjects_df[(all_subjects_df.quiz_class != "relational")]

human_idx = str(all_subjects_df["subject_type"].unique().tolist().index("human"))

# %% Begin modeling

i = 1
print("Non-interacted model:")

res_subjplusclass = smf.logit(
    formula=f"respondent_scores ~ C(subject_type, Treatment(reference={human_idx}))+ C(quiz_class, Treatment(reference={i}))",
    data=all_subjects_df,
).fit(maxiter=1000, method="bfgs")

print(res_subjplusclass.summary())

print("Interacted model:")

res_subjXclass = smf.logit(
    formula=f"respondent_scores ~ C(subject_type, Treatment(reference={human_idx}))*C(quiz_class, Treatment(reference={i}))",
    data=all_subjects_df,
).fit(maxiter=1000, method="bfgs")

print(res_subjXclass.summary())

degfree = 4

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

quiz_class_inclusion_list = ["categorial"]

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

quiz_class_inclusion_list = ["multi_attribute"]

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

quiz_class_inclusion_list = ["numeric"]

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

quiz_class_inclusion_list = ["numeric_multi_attribute"]

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

quiz_class_inclusion_list = ["relational"]

specific_conditions_df = all_subjects_df_with_relational[
    all_subjects_df_with_relational.quiz_class.isin(quiz_class_inclusion_list)
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


# for a given subject, within categorial / multi attribute, what is the effect of condition with reference categorial?

# human subjects

categorial_idx = str(
    all_subjects_df["quiz_class"].unique().tolist().index("categorial")
)

quiz_class_inclusion_list = ["categorial", "multi_attribute"]

specific_conditions_df = all_subjects_df_with_relational[
    all_subjects_df_with_relational.quiz_class.isin(quiz_class_inclusion_list)
]

specific_conditions_df = specific_conditions_df[
    all_subjects_df_with_relational.subject_type.isin(["human"])
]

print(
    "Effect of condition with only the conditions "
    + str(quiz_class_inclusion_list)
    + " for human subjects:"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(quiz_class, Treatment(reference="
    + categorial_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

# model

quiz_class_inclusion_list = ["categorial", "multi_attribute"]

specific_conditions_df = all_subjects_df_with_relational[
    all_subjects_df_with_relational.quiz_class.isin(quiz_class_inclusion_list)
]

specific_conditions_df = specific_conditions_df[
    ~all_subjects_df_with_relational.subject_type.isin(["human"])
]

print(
    "Effect of condition with only the conditions "
    + str(quiz_class_inclusion_list)
    + " for model:"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(quiz_class, Treatment(reference="
    + categorial_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())


# for a given subject, within numeric / numeric multi attribute, what is the effect of condition with reference numeric?

# human subjects

numeric_idx = str(all_subjects_df["quiz_class"].unique().tolist().index("categorial"))

quiz_class_inclusion_list = ["numeric", "numeric_multi_attribute"]

specific_conditions_df = all_subjects_df_with_relational[
    all_subjects_df_with_relational.quiz_class.isin(quiz_class_inclusion_list)
]

specific_conditions_df = specific_conditions_df[
    all_subjects_df_with_relational.subject_type.isin(["human"])
]

print(
    "Effect of condition with only the conditions "
    + str(quiz_class_inclusion_list)
    + " for human subjects:"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(quiz_class, Treatment(reference="
    + numeric_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())

# model

numeric_idx = str(all_subjects_df["quiz_class"].unique().tolist().index("categorial"))

quiz_class_inclusion_list = ["numeric", "numeric_multi_attribute"]

specific_conditions_df = all_subjects_df_with_relational[
    all_subjects_df_with_relational.quiz_class.isin(quiz_class_inclusion_list)
]

specific_conditions_df = specific_conditions_df[
    ~all_subjects_df_with_relational.subject_type.isin(["human"])
]

print(
    "Effect of condition with only the conditions "
    + str(quiz_class_inclusion_list)
    + " for model:"
)

res_subj = smf.logit(
    formula="respondent_scores ~ C(quiz_class, Treatment(reference="
    + numeric_idx
    + "))",
    data=specific_conditions_df,
).fit(maxiter=1000, method="bfgs")

print(res_subj.summary())
