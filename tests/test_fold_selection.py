import unittest

from utils.fold_selection import resolve_fold_indices


class ResolveFoldIndicesTests(unittest.TestCase):
    def test_none_selects_all_folds(self):
        self.assertEqual(
            resolve_fold_indices(None, 4),
            (1, 2, 3, 4),
        )

    def test_valid_fold_selects_one_split(self):
        self.assertEqual(
            resolve_fold_indices(3, 10),
            (3,),
        )

    def test_out_of_range_fold_is_rejected(self):
        for held_out_fold in (-1, 0, 11):
            with self.subTest(held_out_fold=held_out_fold):
                with self.assertRaisesRegex(
                    ValueError,
                    "held_out_fold must be between 1 and 10",
                ):
                    resolve_fold_indices(held_out_fold, 10)

    def test_non_integer_fold_is_rejected(self):
        for held_out_fold in (True, False, 3.0, "3"):
            with self.subTest(held_out_fold=held_out_fold):
                with self.assertRaisesRegex(
                    TypeError,
                    "held_out_fold must be an integer or null",
                ):
                    resolve_fold_indices(held_out_fold, 10)


if __name__ == "__main__":
    unittest.main()
