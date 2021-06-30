An optimization program to maximize 'AvgPointsPerGame' of a DraftKings lineup, specifically for PGA tournaments. In order to execute the script, you will need to download roster csv from DraftKings website for the specific tournament.

Originally tried to use scipy.optimize library, but the methods needed for lineup optimization does not support constraints. For now, using a crude optimization based on high volume iterations (i.e. guess and check... a lot)
