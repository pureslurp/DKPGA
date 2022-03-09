Repo is organized by course not tournament! Course attributes have been included in the script and should be back tested upon completion of each tournament for validity.

Current model attributes:
- Weight DK PPG 5 to 20 pts 
- Weight hole yardage efficiency 0 to 2 pts per hole (https://www.pgatour.com/stats/categories.RSCR_INQ.html)
- Weight course fit based on datagolf.com (https://datagolf.com/course-fit-tool) Total Adjustment * 75
- Past scores from last year and following year tournaments 
- Betting odds 0 to 7.5
- Optional add ons based on course attributes (See commented code in V2 script)

How to interpret the outputs:
Each tournament folder has a csv file that contains,
- DKData: This CSV contains the details of each attribute the model is accessing
- DKFinal: This CSV is the final scoring that is assigned to each player based on the model (sum of the previous csv -- no details)
- DKSalaries-X: The exported CSV from DraftKings
- Maximized_Lineups: The CSV that has the lineups that I enter into DraftKings contests
- Player_Ownership: The percentage ownership of each player from the maximized lineups
