from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

'''
Script that scrapes the ESPN schedule page to get the tournament name and course name

Output: 
- prints TOURNAMENT_LIST_202X dictionary to be put in utils.py
'''


# Set up the WebDriver for Firefox
driver = webdriver.Firefox()

# Open the target URL
url = "https://www.espn.com/golf/schedule"
driver.get(url)

try:
    # Wait for the page to load by checking for the presence of tournament entries
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.eventAndLocation__innerCell"))
    )

    # Find all tournament entries
    tournament_entries = driver.find_elements(By.CSS_SELECTOR, "div.eventAndLocation__innerCell")

    # Extract data into a dictionary
    tournament_dict = {}
    for entry in tournament_entries:
        # Extract tournament name
        print(entry)
        tournament_name_element = entry.find_element(By.CSS_SELECTOR, "p.eventAndLocation__tournamentLink")
        tournament_course_element = entry.find_element(By.CSS_SELECTOR, "div.eventAndLocation__tournamentLocation")
        tournament_name = tournament_name_element.text.replace(" ", "_") if tournament_name_element else "Unknown_Tournament"
        # Extract course name, example "Kapalua Resort (Plantation Course) - Kapalua, HI" should return "Kapalua_Resort_(Plantation_Course)"
        tournament_course = tournament_course_element.text.split(" - ")[0] if tournament_course_element else "Unknown_Course"
        tournament_course = tournament_course.replace(" ", "_") if tournament_course else "Unknown_Course"

        tournament_dict[tournament_name] = tournament_course

    # Print the results
    print(tournament_dict)

except Exception as e:
    print("Error: ", e)

finally:
    # Close the browser
    driver.quit()
