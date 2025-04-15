import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from utils import fix_names  # Add this import at the top
import os

class DataGolfScraper:
    def __init__(self):
        self.url = "https://datagolf.com/course-fit-tool"
        
    def setup_driver(self):
        """Initialize the Selenium WebDriver with appropriate options"""
        options = webdriver.FirefoxOptions()
        # Uncomment below line to run in headless mode
        # options.add_argument('--headless')
        return webdriver.Firefox(options=options)
    
    def wait_for_table(self, driver):
        """Wait for the data table to load"""
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "table"))
            )
            # Add small delay to ensure data is fully loaded
            time.sleep(2)
        except Exception as e:
            print(f"Error waiting for table: {e}")
            return False
        return True
    
    def parse_player_row(self, row):
        """Parse a single player row and extract relevant data"""
        try:
            # Extract player name
            name_col = row.find('div', class_='name-col')
            name = name_col.find('a', class_='tsg-linker').text.strip()
            
            # Reformat name from "Last First" to "First Last"
            name_parts = name.split()
            if len(name_parts) == 2:
                # Regular case: "Smith John" -> "John Smith"
                name = f"{name_parts[1]} {name_parts[0]}"
            elif len(name_parts) == 3:
                # Three name case: "Lee Min Woo" -> "Min Woo Lee"
                name = f"{name_parts[1]} {name_parts[2]} {name_parts[0]}"
            
            name = fix_names(name)  # Apply fix_names function
            
            # Extract skill values
            columns = row.find_all('div', class_='data')
            
            # Get values from the ev-text elements
            values = []
            for col in columns[2:7]:  # Columns 2-6 contain the numerical data
                ev_text = col.find('div', class_='ev-text')
                if ev_text:
                    value = float(ev_text.text.strip().replace('+', ''))
                    values.append(value)
                else:
                    values.append(None)
            
            return {
                'Player': name,
                'Short Game': values[0],
                'Approach': values[1],
                'Distance': values[2],
                'Accuracy': values[3],
                'Total SG Adjustment': values[4]
            }
            
        except Exception as e:
            print(f"Error parsing row: {e}")
            return None
    
    def scrape_data(self):
        """Main method to scrape data from DataGolf"""
        driver = self.setup_driver()
        try:
            print("Accessing DataGolf course fit tool...")
            driver.get(self.url)
            
            if not self.wait_for_table(driver):
                return None
                
            # Get the page source after dynamic content is loaded
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Find all player rows
            rows = soup.find_all('div', class_='datarow')
            
            # Parse each row
            data = []
            for row in rows:
                player_data = self.parse_player_row(row)
                if player_data:
                    data.append(player_data)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add normalized fit score column
            df['fit score'] = (df['Total SG Adjustment'] - df['Total SG Adjustment'].min()) / (
                df['Total SG Adjustment'].max() - df['Total SG Adjustment'].min()
            )
            
            return df
            
        except Exception as e:
            print(f"Error scraping data: {e}")
            return None
            
        finally:
            driver.quit()
    
    def save_to_csv(self, df, tourney, filename='course_fit_dg.csv'):
        """Save the DataFrame to a CSV file in the tournament directory"""
        if df is not None:
            # Create directory path
            directory = os.path.join('2025', tourney)
            
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Create full file path
            filepath = os.path.join(directory, filename)
            
            # Save the file
            df.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        else:
            print("No data to save")

def main(tourney):
    scraper = DataGolfScraper()
    df = scraper.scrape_data()
    
    if df is not None:
        print("\nFirst few rows of scraped data:")
        print(df.head())
        scraper.save_to_csv(df, tourney)
    else:
        print("Failed to scrape data")

if __name__ == "__main__":
    tourney = "RBC_Heritage"
    main(tourney)