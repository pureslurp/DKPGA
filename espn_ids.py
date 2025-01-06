from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

'''
Script that maps tournament names to espn ids

Output:
- prints TOURNAMENT_IDS_202X dictionary to be put in utils.py
'''

# Set up the WebDriver for Firefox
driver = webdriver.Firefox()

# Open the target URL
url = "https://www.espn.com/golf/schedule/_/season/2025"
driver.get(url)

try:
    # Wait for the page to load by checking for the presence of tournament links
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "a.AnchorLink"))
    )

    # Find all anchor elements with the class 'AnchorLink'
    anchor_elements = driver.find_elements(By.CSS_SELECTOR, "a.AnchorLink")

    # Extract data into a dictionary
    tournament_dict = {}
    for anchor in anchor_elements:
        href = anchor.get_attribute("href")
        if "leaderboard?tournamentId=" in href:
            tournament_id = href.split("tournamentId=")[1]
            tournament_name = anchor.text.replace(" ", "_")
            tournament_dict[tournament_name] = int(tournament_id)

    # Print the results
    print(tournament_dict)

except Exception as e:
    print("Error: ", e)
    print("Page source for debugging:")
    print(driver.page_source)

finally:
    # Close the browser
    driver.quit()
