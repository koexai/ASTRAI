"""Utilities for selecting cross-validation splits."""


def resolve_fold_indices(test_fold, n_splits):
    """Return the cross-validation split indices to execute.

    If ``test_fold`` is None, all outer splits are selected. Otherwise, only
    the split corresponding to the requested final test fold is selected.
    """
    if test_fold is None:
        return tuple(range(1, n_splits + 1))

    if isinstance(test_fold, bool) or not isinstance(test_fold, int):
        raise TypeError("test_fold must be an integer or null")

    if not 1 <= test_fold <= n_splits:
        raise ValueError(
            f"test_fold must be between 1 and {n_splits}, "
            f"found {test_fold}"
        )

    return (test_fold,)
