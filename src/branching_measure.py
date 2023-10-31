"""Calculate the branching of given trees."""
from typing import Generator, TypeAlias

import nltk
import numpy as np

from .mylogger import main_logger

logger = main_logger.getChild(__name__)

#####################
## Tree Definition ##
#####################

SpanTree: TypeAlias = list[
    tuple[int, int]
]  # list of pairs of indices (i, j) expressing the start/end index of the (unlabeld) constituent.
# Note that both SpanTree cannot represent all possible trees.
# E.g., there is not SpanTree for single node without any leaves.

Leaf: TypeAlias = str

DeepListTree: TypeAlias = (
    Leaf | list["DeepListTree"]
)  # This defines unlabeled constituency trees.

##########
## Util ##
##########


def leaves_of_deeplisttree(dlt: DeepListTree) -> Generator[str, None, None]:
    if isinstance(dlt, Leaf):
        yield dlt
    else:
        for element in dlt:
            yield from leaves_of_deeplisttree(element)


def get_num_leaves_deeplisttree(dlt: DeepListTree) -> int:
    """Get the number of leaves."""
    return len(list(leaves_of_deeplisttree(dlt)))


def get_num_leaves_spantree(st: SpanTree) -> int:
    """Get the number of leaves."""
    return max([s[0] for s in st] + [s[1] for s in st]) + 1


def get_num_children_deeplisttree(dlt: DeepListTree) -> int:
    if isinstance(dlt, Leaf):
        return 0
    else:
        return len(dlt)


def spantree2deeplisttree(st: SpanTree) -> DeepListTree:
    """Convert given SpanTree to DeepListTree.

    Note that some SpanTree, e.g., those with crossing constituents, cannot be converted to DeepListTree.
    But we will not check.
    """

    # Obtain the number of leaves:
    num_leaves: int = get_num_leaves_spantree(st)

    # First convert the span tree into string deeplist tree.
    left_paren: str = "["
    right_paren: str = "]"

    bases: list[str] = [f"'{i}'" for i in range(num_leaves)]
    for left_idx, right_idx in st:
        # Process the left index.
        bases[left_idx] = left_paren + bases[left_idx]

        # Process the right index.
        bases[right_idx] = bases[right_idx] + right_paren

    str_deeplisttree: str = ", ".join(bases)

    dlt: DeepListTree = eval(str_deeplisttree)

    return dlt


def deeplist2spantree(dlt: DeepListTree) -> SpanTree:
    if isinstance(dlt, Leaf):
        # Return empty set if l is a leaf.
        return list()

    else:
        # Add the constituent covering all the leaves.
        st: SpanTree = [(0, get_num_leaves_deeplisttree(dlt) - 1)]

        num_leaves_from_left: int = 0
        for child in dlt:
            child_num_leaves: int = get_num_leaves_deeplisttree(child)

            for left_idx, right_idx in deeplist2spantree(child):
                # Offset indices by the sum of number of leaves of the left siblings of this child.
                new_left_idx: int = left_idx + num_leaves_from_left
                new_right_idx: int = right_idx + num_leaves_from_left

                st.append((new_left_idx, new_right_idx))

            num_leaves_from_left += child_num_leaves

        return st


def nltktree2deeplisttree(nltkt: nltk.Tree) -> DeepListTree:
    """Remove labels from PTB-like trees."""
    if isinstance(nltkt, str):
        # Leaf.
        return Leaf(nltkt)
    else:
        dlt: DeepListTree = []
        for child in nltkt:
            child_dlt: DeepListTree = nltktree2deeplisttree(child)
            dlt.append(child_dlt)
        return dlt


def rename_sub(t: DeepListTree, next_id: int) -> tuple[DeepListTree, int]:
    """Rename the leaves by indecies, e.g., 0, 1, 2, ..."""
    if isinstance(t, Leaf):
        return Leaf(next_id), next_id + 1
    else:
        renamed_t: DeepListTree = []
        for child in t:
            renamed_child, next_id = rename_sub(child, next_id)

            renamed_t.append(renamed_child)
        return renamed_t, next_id


def rename_deeplisttree(dlt: DeepListTree) -> DeepListTree:
    """Rename the leaves by depth-first traversal from the left."""
    renamed_dlt, _ = rename_sub(dlt, 0)
    return renamed_dlt


def flip_deeplisttree(dlt: DeepListTree) -> DeepListTree:
    if isinstance(dlt, Leaf):
        return dlt
    else:
        flipped_dlt: DeepListTree = []

        for child in dlt[::-1]:
            flipped_child: DeepListTree = flip_deeplisttree(child)
            flipped_dlt.append(flipped_child)
        return flipped_dlt


def iterate_over_nodes(dlt: DeepListTree) -> Generator[DeepListTree, None, None]:
    """Bottom-up iteration."""

    # First iterate over descendants.
    if not isinstance(dlt, Leaf):
        for child in dlt:
            yield from iterate_over_nodes(child)

    # Yield itself.
    yield dlt


#######################
## Branching Measure ##
#######################


def round_to_infinite(x: float) -> int:
    """Round a given value so that it has a larger abs."""
    return np.sign(x) * np.ceil(np.abs(x))


def calc_children_weights(num_children: int) -> list[float]:
    """Calculate the weights for children.

    If num_children=5, the weights are (-1, -1/2, 0, 1/2, 1).
    The weight is 0 for unary node.
    When a node is a leaf, the weights are not defined.
    """
    if num_children <= 1:
        return [0.0 for _ in range(num_children)]
    else:
        step: float = 1 / (np.floor(num_children / 2))

        weights: list[float] = []
        for i in range(num_children):
            # Calculate the index of children by setting the center as 0.
            # If there are 5 children, then (-2, -1, 0, 1, 2)
            # If there are 4, then (-2, -1, 1, 2)
            index_from_center: int = round_to_infinite(i - ((num_children - 1) / 2))
            weight: float = index_from_center * step

            weights.append(weight)

        return weights


def weighted_relative_difference(dlt: DeepListTree) -> float:
    """Calculate the relative difference in the number of leaves of the direct children of the root.

    Although the difference is not defined for leaf nodes, this function returns 0.0 for simplicity.
    """

    weights: list[float] = calc_children_weights(get_num_children_deeplisttree(dlt))
    child_num_leaves: list[int] = (
        [get_num_leaves_deeplisttree(child) for child in dlt]
        if not isinstance(dlt, Leaf)
        else []
    )

    return np.sum([w * n for w, n in zip(weights, child_num_leaves)])


def nary_relative_corrected_colles_index(dlt: DeepListTree) -> float:
    # First, calculate (unnomalized) Colles index.
    # Note that the relative difference is always 0 for leaves so that it doesn't affect the value.
    nr_colles_index: float = np.sum(
        [weighted_relative_difference(v) for v in iterate_over_nodes(dlt)]
    )

    num_leaves: int = get_num_leaves_deeplisttree(dlt)
    nr_corrected_colles_index: float = (2 * nr_colles_index) / (
        (num_leaves - 1) * (num_leaves - 2)
    )

    return nr_corrected_colles_index


def nary_relative_equal_weights_corrected_colles_index(dlt: DeepListTree) -> float:
    # Note that the relative difference is always 0 for leaves so that it doesn't affect the value.

    sum_normalized_diff: float = 0
    for v in iterate_over_nodes(dlt):
        num_leaves_v: int = get_num_leaves_deeplisttree(v)

        if num_leaves_v > 2:
            sum_normalized_diff += weighted_relative_difference(v) / (num_leaves_v - 2)

    num_leaves: int = get_num_leaves_deeplisttree(dlt)
    nr_equal_weights_colles_index: float = sum_normalized_diff / (num_leaves - 2)

    return nr_equal_weights_colles_index


def nary_relative_rogers_j_index(dlt: DeepListTree) -> float:
    # First, calculate (unnomalized) Colles index.
    # Note that the relative difference is always 0 for leaves so that it doesn't affect the value.
    sum_sign: float = np.sum(
        [np.sign(weighted_relative_difference(v)) for v in iterate_over_nodes(dlt)]
    )

    num_leaves: int = get_num_leaves_deeplisttree(dlt)
    nr_staircaseness: float = sum_sign / (num_leaves - 2)

    return nr_staircaseness


##########
## Test ##
##########
if __name__ == "__main__":
    # Right branching
    rt: DeepListTree = ["0", ["1", ["2", ["3", "4"]]]]

    # Left branching
    lt: DeepListTree = [[[["0", "1"], "2"], "3"], "4"]

    # Complete binary
    bt: DeepListTree = [[["0", "1"], ["2", "3"]], [["4", "5"], ["6", "7"]]]

    # Non-binary
    nbt: DeepListTree = [[["0"], ["1", "2", "3"]], ["4", "5"], [["6", "7", "8"]]]
    nbt_flip: DeepListTree = [[["0", "1", "2"]], ["3", "4"], [["5", "6", "7"], ["8"]]]

    nbt2: DeepListTree = [["0", [["1", "2"], "3"]], ["4", ["5", "6"]], [["7"]]]

    trees: list[DeepListTree] = [rt, lt, bt, nbt, nbt2]

    # Check flip and rename.
    flipped_rt: DeepListTree = rename_deeplisttree(flip_deeplisttree(rt))
    assert repr(flipped_rt) == repr(lt)

    flipped_lt: DeepListTree = rename_deeplisttree(flip_deeplisttree(lt))
    assert repr(flipped_lt) == repr(rt)

    flipped_bt: DeepListTree = rename_deeplisttree(flip_deeplisttree(bt))
    assert repr(flipped_bt) == repr(bt)

    flipped_nbt: DeepListTree = rename_deeplisttree(flip_deeplisttree(nbt))
    assert repr(flipped_nbt) == repr(nbt_flip)

    # Check conversion.
    for t in trees:
        s: SpanTree = deeplist2spantree(t)
        tt: DeepListTree = spantree2deeplisttree(s)
        ss: SpanTree = deeplist2spantree(tt)

        assert repr(t) == repr(rename_deeplisttree(tt))
        assert repr(s) == repr(ss)

    # Check branching measure.

    assert repr(calc_children_weights(5)) == repr([-1.0, -1 / 2, 0.0, 1 / 2, 1.0])
    assert repr(calc_children_weights(6)) == repr(
        [-1.0, -2 / 3, -1 / 3, 1 / 3, 2 / 3, 1.0]
    )

    assert weighted_relative_difference(rt) == (4 - 1)
    assert weighted_relative_difference(lt) == (1 - 4)
    assert weighted_relative_difference(bt) == 0
    assert weighted_relative_difference(nbt) == -1
    assert weighted_relative_difference(nbt_flip) == 1

    def check_branching_measure(b):
        assert b(rt) == 1.0
        assert b(lt) == -1.0
        assert b(bt) == 0.0

        for tree in trees:
            assert b(tree) == -b(flip_deeplisttree(tree))

    check_branching_measure(nary_relative_corrected_colles_index)
    check_branching_measure(nary_relative_equal_weights_corrected_colles_index)
    check_branching_measure(nary_relative_rogers_j_index)

    assert nary_relative_corrected_colles_index(nbt2) == -1 / 21
    assert nary_relative_equal_weights_corrected_colles_index(nbt2) == 1 / 12
    assert nary_relative_rogers_j_index(nbt2) == 0.0
