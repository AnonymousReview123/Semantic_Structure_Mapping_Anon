import csv
import random
from typing import List, Tuple

import numpy as np
from numpy import typing as npt

import grounding_generation as gg

analogies = np.array(
    [
        [["dog", "puppy"], ["cat", "kitten"]],
        [["king", "queen"], ["man", "woman"]],
        [["square", "rectangle"], ["circle", "oval"]],
        [["black", "gray"], ["red", "pink"]],
    ]
)
ANALOGIES = analogies[:4]
RANDOM_WORDS = np.array(
    [
        [["gun", "yellow"], ["darkness", "carpet"]],
        [["zebra", "bullet"], ["lightbulb", "book"]],
        [["banana", "fireplace"], ["bean", "plug"]],
        [["wire", "beer"], ["duck", "wife"]],
    ]
)
GROUNDINGS = np.array(
    [
        gg.generate_groundings("C C C", gg.lower_case, gg.change_count(-1)),
        gg.generate_groundings("} ^ {", gg.extract_center, gg.replace("^", "*")),
        gg.generate_groundings("Q Z I", gg.fine_grained_duplication, gg.reduplicate),
        gg.generate_groundings("E K C", gg.interstitial_insertion("%"), gg.reverse),
    ]
)
DISTRACTORS = ["door", "lime", "tree", "pillow"]
DISTRACTOR_GROUNDINGS = ["Y Y Y", "8 8", "A P", "U"]

SEPARATOR = " => "  # This is displayed between a given lexical item and grounding in a question.


def generate_question(
    analogy_set: npt.NDArray,
    grounding_set: npt.NDArray,
    flip: bool = False,
    index_ordering: List[int] = None,
    distractor: Tuple[Tuple[str, str], int] = None,
) -> Tuple[str, str]:
    """
    Takes an array of words that form analogies and an array of groundings
    that have an analogy-like pattern and creates a question using them.
    """
    if analogy_set.shape != grounding_set.shape:
        raise ValueError(
            f"Analogy and groundings has mismatching shapes: {analogy_set.shape} {grounding_set.shape}"
        )
    pairs = [
        (word, grounding)
        for word, grounding in zip(analogy_set.flatten(), grounding_set.flatten())
    ]
    if index_ordering:
        pairs = [pairs[i] for i in index_ordering]
    if distractor:
        pairs = pairs[: distractor[1]] + [distractor[0]] + pairs[distractor[1] :]
    if flip:
        pairs = [(pair[1], pair[0]) for pair in pairs]
    return (
        "".join([f"{pair[0]}{SEPARATOR}{pair[1]}\n" for pair in pairs[:-1]])
        + f"{pairs[-1][0]}{SEPARATOR}"
    ), pairs[-1][1]


def convert_to_only_rhs(question):
    question_words = []

    line_words = []

    question_lines = question.splitlines()

    for line in question_lines:
        line_words = line.split(" => ")
        for word in line_words:
            question_words.append(word)

    question_words_rhs = []
    question_words_rhs.append(question_words[1])
    question_words_rhs.append(question_words[3])
    question_words_rhs.append(question_words[5])

    new_question = ""

    for word in question_words_rhs:
        new_question += word + "\n"

    new_question = new_question[:-1]  # remove the last \n

    return new_question


def generate_quiz(
    analogies: npt.ArrayLike,
    groundings: npt.ArrayLike,
    flips: List[bool] = None,
    orderings: List[List[int]] = None,
    distractors: List[Tuple[Tuple[str, str], int]] = None,
) -> Tuple[List[str], List[str]]:
    """Takes the information required to create a quiz and returns lists of questions and answers for that quiz."""
    questions = []
    answers = []
    for i in range(len(analogies)):
        analogy_set, grounding_set = analogies[i], groundings[i]
        flip = flips[i] if flips else None
        ordering = orderings[i] if orderings else None
        distractor = distractors[i] if distractors else None
        question, answer = generate_question(
            analogy_set,
            grounding_set,
            flip=flip,
            index_ordering=ordering,
            distractor=distractor,
        )
        questions.append(question)
        answers.append(answer)
    return questions, answers


def generate_pilot() -> Tuple[
    List[Tuple[List[str], List[str]]],
    List[Tuple[List[str], List[str]]],
    List[Tuple[List[str], List[str]]],
    List[Tuple[List[str], List[str]]],
    List[Tuple[List[str], List[str]]],
    List[Tuple[List[str], List[str]]],
]:
    """Generates and returns all quizzes for the pilot study."""

    ag_pairings: List[List[List[int]]] = []
    pairing_bank = [(i, j) for i in range(4) for j in range(4)]

    while len(pairing_bank) != 0:
        ag_pairings.append([[], []])
        curr = ag_pairings[-1]
        while len(curr[0]) != 4:
            if len(curr[0]) == 3:  # at this point, the final pairing is forced
                needed = [
                    (i, j)
                    for i in range(4)
                    if i not in curr[0]
                    for j in range(4)
                    if j not in curr[1]
                ][0]
                if needed not in pairing_bank:  # trapped -> restart
                    ag_pairings: List[List[List[int]]] = []
                    pairing_bank = [(i, j) for i in range(4) for j in range(4)]
                    break
                else:  # efficient completion
                    curr[0].append(needed[0])
                    curr[1].append(needed[1])
                    pairing_bank.remove(needed)
                    break
            new = random.choice(pairing_bank)
            if new[0] in curr[0] or new[1] in curr[1]:
                continue
            curr[0].append(new[0])
            curr[1].append(new[1])
            pairing_bank.remove(new)
        if len(pairing_bank) == 0:
            if (
                len(set([tuple(pair[0]) for pair in ag_pairings])) != 4
                or len(set([tuple(pair[1]) for pair in ag_pairings])) != 4
            ):
                ag_pairings = []
                pairing_bank = [(i, j) for i in range(4) for j in range(4)]

    contentful_pairings = [
        (ANALOGIES[pairing[0]], GROUNDINGS[pairing[1]]) for pairing in ag_pairings
    ]

    defaults = [
        generate_quiz(analogies, groundings)
        for analogies, groundings in contentful_pairings
    ]

    question_orderings: List[List[int]] = [[x for x in range(4)] for _ in range(4)]
    for i in range(4):
        random.shuffle(question_orderings[i])

    permuted_questions = [
        generate_quiz(
            [analogies[i] for i in ordering], [groundings[i] for i in ordering]
        )
        for (analogies, groundings), ordering in zip(
            contentful_pairings, question_orderings
        )
    ]

    distractors: List[List[Tuple[Tuple[str, str], int]]] = []
    for _ in range(4):
        random.shuffle(DISTRACTORS)
        random.shuffle(DISTRACTOR_GROUNDINGS)
        insertion_points = list(range(4))
        random.shuffle(insertion_points)
        distractors.append(
            [
                ((d, d_g), i)
                for d, d_g, i in zip(
                    DISTRACTORS, DISTRACTOR_GROUNDINGS, insertion_points
                )
            ]
        )

    distracted = [
        generate_quiz(analogies, groundings, distractors=d)
        for (analogies, groundings), d in zip(contentful_pairings, distractors)
    ]

    pair_orderings: List[List[List[int]]] = [
        [[x for x in range(4)] for _ in range(4)] for _ in range(4)
    ]
    for i in range(4):
        while (
            len(set(tuple(l) for l in pair_orderings[i])) != 4
        ):  # use unique scramble for each question within a quiz
            for j in range(4):
                random.shuffle(pair_orderings[i][j])

    permuted_pairs = [
        generate_quiz(analogies, groundings, orderings=o)
        for (analogies, groundings), o in zip(contentful_pairings, pair_orderings)
    ]

    random_contentful_pairings = [
        (RANDOM_WORDS[pairing[0]], GROUNDINGS[pairing[1]]) for pairing in ag_pairings
    ]

    randoms = [
        generate_quiz(analogies, groundings)
        for analogies, groundings in random_contentful_pairings[:2]
    ]

    random_permuted_pairs = [
        generate_quiz(analogies, groundings, orderings=o)
        for (analogies, groundings), o in zip(
            random_contentful_pairings[2:], pair_orderings[2:]
        )
    ]

    return (
        defaults,
        distracted,
        permuted_questions,
        permuted_pairs,
        randoms,
        random_permuted_pairs,
    )


def write_list_to_csv(l: List, path: str):
    """Writes a list to the first column of a csv file given a path to write to."""
    with open(f"{path}", "w", newline="", encoding="UTF8") as f:
        writer = csv.writer(f)
        for elem in l:
            writer.writerow([elem])

pilot_path = "quiz_files/phase_1/"

# generate random_finals

# for i in range(1, 5):
#     path_to_defaults = pilot_path + f"defaults/defaults{i}/"
#     questions_path = path_to_defaults + f"defaults{i}_questions.csv"
#     answers_path = path_to_defaults + f"defaults{i}_answers.csv"
#     questions = []

#     with open(questions_path, "r") as questions_file:
#         csvreader = csv.reader(questions_file)
#         for row in csvreader:
#             questions.append(row)

#     random_words = random.sample(DISTRACTORS, len(DISTRACTORS))
#     random_final_questions = []
#     for j in range(len(questions)):
#         lines = questions[j][0].split("\n")
#         lines[-1] = (
#             random_words[j] + SEPARATOR + lines[-1].split(SEPARATOR)[1]
#         )
#         random_final_questions.append("\n".join(lines))

#     quiz_path = f"{pilot_path}random_finals/random_finals{i}/"
#     os.makedirs(quiz_path)
#     shutil.copyfile(answers_path, f"{quiz_path}random_finals{i}_answers.csv")
#     write_list_to_csv(
#         random_final_questions, f"{quiz_path}random_finals{i}_questions.csv"
#     )


# # generate onlyRHS

# for i in range(1, 5):
#     path_to_defaults = pilot_path + f"defaults/defaults{i}/"
#     questions_path = path_to_defaults + f"defaults{i}_questions.csv"
#     answers_path = path_to_defaults + f"defaults{i}_answers.csv"
#     questions = []

#     with open(questions_path, "r") as questions_file:
#         csvreader = csv.reader(questions_file)
#         for row in csvreader:
#             questions.append(row)

#     # random_words = random.sample(DISTRACTORS, len(DISTRACTORS))
#     onlyrhs_questions = []
#     for j in range(len(questions)):
#         onlyrhs_questions.append(convert_to_only_rhs(questions[j][0]))

#     quiz_path = f"{pilot_path}only_rhs/only_rhs{i}/"
#     os.makedirs(quiz_path)
#     shutil.copyfile(answers_path, f"{quiz_path}only_rhs{i}_answers.csv")
#     write_list_to_csv(onlyrhs_questions, f"{quiz_path}only_rhs{i}_questions.csv")
