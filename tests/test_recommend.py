import unittest

try:
    import pandas as pd
except Exception:
    pd = None

class RecommendSweetSpotTest(unittest.TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest('pandas not available')

    def test_recommend_selects_min_energy_row(self):
        from recommend import recommend_sweet_spot
        df = pd.DataFrame({
            'max_watt': [150, 200, 250],
            'energy_consumption_watt_min': [100.0, 80.0, 120.0]
        }).set_index('max_watt')

        result = recommend_sweet_spot(df)
        self.assertEqual(result, 200)

if __name__ == '__main__':
    unittest.main()
