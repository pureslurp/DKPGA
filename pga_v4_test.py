import unittest
import pandas as pd
from pga_v4 import LineUp, Golfer

TEST_DATA = pd.read_csv('test_utils/dk_final_test.csv')
TEST_LINEUPS = pd.read_csv('test_utils/dk_lineups_test.csv')


class TestOptimization(unittest.TestCase):
    def test_optimization(self):
        lineup = TEST_LINEUPS.iloc[0]
        lineup_obj = LineUp(Golfer(TEST_DATA[TEST_DATA["Name + ID"] == lineup["G1"]]), Golfer(TEST_DATA[TEST_DATA["Name + ID"] == lineup["G2"]]),  Golfer(TEST_DATA[TEST_DATA["Name + ID"] == lineup["G3"]]),  Golfer(TEST_DATA[TEST_DATA["Name + ID"] == lineup["G4"]]),  Golfer(TEST_DATA[TEST_DATA["Name + ID"] == lineup["G5"]]), Golfer(TEST_DATA[TEST_DATA["Name + ID"] == lineup["G6"]]))
        print(lineup_obj.optimize(TEST_DATA))

    
if __name__ == '__main__':
    unittest.main()