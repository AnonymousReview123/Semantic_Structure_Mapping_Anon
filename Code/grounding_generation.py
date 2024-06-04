from collections.abc import Callable
from typing import Tuple

import numpy as np
from numpy import typing as npt

# * ASSUMPTION: These fxns assume groundings are always characters with spaces between


def check_distinct(grounding: str, transformed: str, description: str):
    """Checks whether the grounding and the result when it is transformed are distinct and raises an error using the description if not."""
    if grounding == transformed:
        raise ValueError(
            f"{description} does not result in a noticeable change when applied to {grounding}."
        )


def lower_case(grounding: str) -> str:
    """Returns the lower case version of an upper case grounding."""
    if not grounding.isupper():
        raise Exception(f"{grounding} is not upper case before being lower cased.")
    return grounding.lower()


def remove_first(grounding: str) -> str:
    """Returns the grounding minus the first character."""
    if len(grounding) < 3:
        raise Exception(
            f"{grounding} is too short (removing the first character results in the empty string)."
        )
    return grounding[2:]


def remove_last(grounding: str) -> str:
    """Returns the grounding minus the last character."""
    if len(grounding) < 3:
        raise Exception(
            f"{grounding} is too short (removing the final character results in the empty string)."
        )
    return grounding[:-2]


def rotate(grounding: str) -> str:
    """Returns the grounding with the first character moved to the end."""
    characters = grounding.split(" ")
    transformed = " ".join(characters[1:] + characters[0:1])
    check_distinct(grounding, transformed, "Rotation")
    return transformed


def reduplicate(grounding: str) -> str:
    """Returns the grounding repeated (i.e., two copies of the grounding as one string)."""
    return grounding + " " + grounding


def fine_grained_duplication(grounding: str) -> str:
    """Returns the grounding with each character repeated after itself."""
    characters = grounding.split(" ")
    return " ".join([c + " " + c for c in characters])


def reverse(grounding: str) -> str:
    """Returns the grounding with characters in reverse order."""
    reversed = grounding[::-1]
    check_distinct(grounding, reversed, "Reversal")
    return reversed


def extract_center(grounding: str) -> str:
    """Returns the character at the center of the inputted grounding."""
    characters = grounding.split(" ")
    if len(characters) % 2 == 0:
        raise ValueError(
            f"{grounding} has an even number of characters. Extracting the center requires a center."
        )
    center = characters[len(characters) // 2]
    check_distinct(grounding, center, "Extracting the center")
    return center


def replace(to_replace: str, replace_with: str) -> Callable[[str], str]:
    """Returns a function that replaces instances of to_replace in a grounding with replace_with using python .replace()."""

    def replacer(grounding: str) -> str:
        replaced = grounding.replace(to_replace, replace_with)
        check_distinct(
            grounding, replaced, f"Replacement of {to_replace} with {replace_with}"
        )
        return replaced

    return replacer


def surround(left: str, right: str = None) -> Callable[[str], str]:
    """Returns a function that surrounds a grounding with left and right, or left on both sides if only one string is provided."""

    def surrounder(grounding: str) -> str:
        if right:
            surrounded = f"{left} {grounding} {right}"
        else:
            surrounded = f"{left} {grounding} {left}"
        check_distinct(
            grounding,
            surrounded,
            f"Surrounding with {left} and {right if right else left}",
        )
        return surrounded

    return surrounder


def interstitial_insertion(insertion_character: str) -> Callable[[str], str]:
    """Returns a function that inserts the insertion_character between every (non-whitespace) character in a grounding, with whitespaces between every (non-whitespace) character."""
    if len(insertion_character) != 1:
        raise ValueError(f"{insertion_character} is not length 1.")

    def inserter(grounding: str) -> str:
        output = f" {insertion_character} ".join(grounding.split(" "))
        check_distinct(
            grounding, output, f"Interstitial insertion of {insertion_character}"
        )
        return output

    return inserter


def change_count(n: int) -> Callable[[str], str]:
    """Returns a function that takes a grounding with a single repeated character and changes the number of times the single character appears by n."""

    def count_changer(grounding: str) -> str:
        characters = grounding.split(" ")
        character_set = set(characters)
        if len(character_set) != 1:
            raise ValueError(
                f"{grounding} is not composed of a single character repeated some number of times with spaces in between."
            )
        length = len(characters) + n
        if length < 1:
            raise Exception(f"{grounding} is too short to have length changed by {n}.")
        return " ".join([characters[0] for i in range(length)])

    return count_changer


def is_valid_grounding(s: str) -> bool:
    """Returns True if s is a valid grounding and false otherwise."""
    if s == "" or len(s) % 2 == 0:
        return False
    for i in range(len(s)):
        if i % 2 == 1 and s[i] != " ":
            return False
    return True


def generate_groundings(
    seed: str,
    row_function: Callable[[str], str],
    col_function: Callable[[str], str],
    shape: Tuple[int] = (2, 2),
) -> npt.NDArray:
    """
    Takes a seed grounding string, a function to apply across rows, and a function to apply down columns,
    and returns a 2D numpy array of groundings of the provided shape, or (2, 2) if no shape is provided.
    """
    if len(shape) != 2:
        raise ValueError(
            f"The shape of a grounding array must be 2 dimensional. {shape} has {len(shape)} dimensions."
        )
    elif not is_valid_grounding(seed):
        raise ValueError(f"{seed} is not a valid grounding.")
    row_based = [[seed]]
    for _ in range(shape[1] - 1):
        row_based[0].append(row_function(row_based[0][-1]))
    for _ in range(shape[0] - 1):
        row_based.append([col_function(x) for x in row_based[-1]])

    col_based = [[seed]]
    for _ in range(shape[0] - 1):
        col_based.append([col_function(col_based[-1][0])])
    for row in range(shape[0]):
        for _ in range(shape[1] - 1):
            col_based[row].append(row_function(col_based[row][-1]))

    row_based = np.array(row_based)
    col_based = np.array(col_based)

    if not np.array_equal(row_based, col_based):
        raise RuntimeError(
            f"Applying the row and column functions in different orders produces mismatched results:\n{row_based}\n{col_based}"
        )
    if len(np.unique(row_based)) != np.size(row_based):
        raise RuntimeError(f"Functions produced duplicate groundings.")

    return row_based
