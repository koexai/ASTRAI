"""Utilities for selecting cross-validation splits."""


def resolve_fold_indices(held_out_fold, n_splits):
    """Return the cross-validation split indices to execute.

    If ``held_out_fold`` is None, all splits are selected. Otherwise, only
    the split corresponding to the requested held-out fold is selected.
    """
    if held_out_fold is None:
        return tuple(range(1, n_splits + 1))

    if isinstance(held_out_fold, bool) or not isinstance(held_out_fold, int):
        raise TypeError("held_out_fold must be an integer or null")

    if not 1 <= held_out_fold <= n_splits:
        raise ValueError(
            f"held_out_fold must be between 1 and {n_splits}, "
            f"found {held_out_fold}"
        )

    return (held_out_fold,)
