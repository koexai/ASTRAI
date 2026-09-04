import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.array_dtypes import (
    INDEX_ARRAY_DTYPE,
    MODEL_ARRAY_DTYPE,
    as_index_array,
    as_model_array,
    load_model_array,
)


class ArrayDtypeContractTests(unittest.TestCase):
    def test_normalises_model_arrays_to_float32(self):
        source = np.array([[1.25, 2.5]], dtype=np.float64)

        result = as_model_array(source)

        self.assertEqual(result.dtype, MODEL_ARRAY_DTYPE)
        np.testing.assert_array_equal(result, source.astype(np.float32))

    def test_reuses_model_arrays_that_already_match_the_contract(self):
        source = np.array([1.0, 2.0], dtype=np.float32)

        result = as_model_array(source)

        self.assertIs(result, source)

    def test_rejects_non_real_model_arrays(self):
        for source in (
            np.array(["not numeric"]),
            np.array([1.0 + 2.0j]),
            np.array([True]),
        ):
            with self.subTest(dtype=source.dtype):
                with self.assertRaisesRegex(TypeError, "real numeric"):
                    as_model_array(source)

    def test_normalises_index_arrays_to_int64(self):
        source = np.array([1, 3, 5], dtype=np.int32)

        result = as_index_array(source)

        self.assertEqual(result.dtype, INDEX_ARRAY_DTYPE)
        np.testing.assert_array_equal(result, source)

    def test_rejects_non_integer_index_arrays(self):
        with self.assertRaisesRegex(TypeError, "must contain integers"):
            as_index_array(np.array([1.0, 2.0], dtype=np.float32))

    def test_loads_legacy_float64_arrays_as_float32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.npy"
            source = np.array([[0.1, 0.2]], dtype=np.float64)
            np.save(path, source)

            result = load_model_array(path)

        self.assertEqual(result.dtype, MODEL_ARRAY_DTYPE)
        np.testing.assert_array_equal(result, source.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
