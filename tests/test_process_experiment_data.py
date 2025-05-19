import unittest

try:
    import pandas as pd
except Exception:
    pd = None

class MergeDataTest(unittest.TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest('pandas not available')

    def test_merge_with_tolerance(self):
        from process_experiment_data import merge_data, MERGE_TOLERANCE
        main_df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2023-01-01 00:00:03'),
                pd.Timestamp('2023-01-01 00:00:10'),
                pd.Timestamp('2023-01-01 00:00:11')
            ],
            'value': [1, 2, 3]
        })

        additional_df = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2023-01-01 00:00:00'),
                pd.Timestamp('2023-01-01 00:00:05')
            ],
            'extra': ['a', 'b']
        })

        merged = merge_data(main_df, additional_df, tolerance=MERGE_TOLERANCE)

        self.assertEqual(merged.loc[0, 'extra'], 'a')
        self.assertEqual(merged.loc[1, 'extra'], 'b')
        self.assertTrue(pd.isna(merged.loc[2, 'extra']))

if __name__ == '__main__':
    unittest.main()
