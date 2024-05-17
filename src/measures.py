import editdistance
from typing import Callable, Iterable, TypeVar, List, Generator

import torch
from typing import List, Tuple, Union, Iterable
from numpy.typing import ArrayLike

# Extra types
Message = Union[str, List[int], torch.Tensor]
Messages = Union[torch.Tensor, List[Message]]

R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")


#################################################
### BEGIN helpers to compute pairwise metrics ###
#################################################

def pairwise(
    f: Callable[[R, S], T], xs: Iterable[R], ys: Iterable[S]
) -> Generator[T, None, None]:
    """Apply f to all combinations of x in xs and y in ys"""
    return itertools.starmap(f, itertools.product(xs, ys))


def pairwise_dedup(f: Callable[[S, S], T], xs: List[S]) -> Generator[T, None, None]:
    """Apply f to all combinations of x1, x2 in xs.
    Triangular without diagonal."""
    for i, a in enumerate(xs[:-1]):
        for b in xs[(i + 1) :]:
            yield f(a, b)

def cumulative_average(xs: Iterable) -> float:
    """Efficient computation of the cumulative average from a generator"""
    ca = 0.0
    for i, x in enumerate(xs):
        ca += (x - ca) / (i + 1)
    return ca

#################################################
### END helpers to compute pairwise metrics   ###
#################################################

def normalized_editdistance(a: Message, b: Message) -> float:
    """Calculates the edit distance divided by maximum length"""
    maxlen = max(len(a), len(b))

    dist = editdistance.eval(a, b)

    try:
        # we normalize by the maximum length of the two sequences
        # such that the distance is in [0, 1] 
        # in contrast to egg which takes average length
        # It doesn't matter for spearman correlation in topsim.
        dist /= maxlen
    except ZeroDivisionError:
        # Both sequences are empty
        return 0.0
    return dist


def production_similarity(a: Message, b: Message) -> float:
    """One minus length-normalized edit distance"""
    return 1 - normalized_editdistance(a, b)



VALID_METRICS = {  
    "editdistance": editdistance.eval,
    "normalized_editdistance": normalized_editdistance,
    "production_similarity": production_similarity,
}
