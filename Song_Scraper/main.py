
# Selenium Chromedriver
from selenium.webdriver import Chrome
# Default Params
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
# Time
import time
# Pandas
import pandas as pd
# Wget
import wget
# SSL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


pages_scrapred = 100
good_bad = 'bad'
pause_time = 5

def scrape_pages():
    # Selenium
    driver = Chrome(default_params['chromedriver_executable'])
    # Resize driver to 720p
    driver.set_window_size(1280, 720)
    # Go to website
    if good_bad == 'good':
        driver.get(default_params['bsaber_good_start_website'])
    else:
        driver.get(default_params['bsaber_bad_start_website'])

    # Pandas Dataframe
    df = pd.DataFrame(columns = ['song_link', 'difficulty', 'good/bad'])

    # Find elements by class:
    time.sleep(pause_time)
    for i in range(pages_scrapred):
        links, next_button_element = get_info_for_page(driver)
        # Add links to dataframe with difficulty and good/bad
        for link in links:
            df = df.append({'song_link' : link, 'difficulty' : default_params['difficulty'], 'good/bad' : good_bad}, ignore_index = True)

        # Click next button
        next_button_element.click()
        # Wait for page to load
        time.sleep(pause_time)

    # Save Dataframe
    if good_bad == 'good':
        df.to_csv(default_params['download_links_csv_good'], index = False)
    else:
        df.to_csv(default_params['download_links_csv_bad'], index = False)

    # Quit driver
    driver.quit()


# Get info for page
def get_info_for_page(driver):
    # Returned
    links = []

    # Get all links and add them to list
    elements = driver.find_elements(by = 'css selector', value = 'a[href*="https://api.beatsaver.com/download/"]')
    for element in elements:
        links.append(element.get_attribute('href'))
    
    # Find next button
    if good_bad == 'good':
        next_button = driver.find_elements(by = 'css selector', value = 'a[href*="https://bsaber.com/songs/top/page/"]')[-1]
    else:
        next_button = driver.find_elements(by = 'css selector', value = 'a[href*="https://bsaber.com/songs/new/page/"]')[-1]

    return links, next_button

# Download Songs
def download_pages_to_training_songs():
    # Download Good Song Zips
    df = pd.read_csv(default_params['download_links_csv_good'])
    for i in range(len(df)):
        try:
            wget.download(df['song_link'][i], f"{default_params['training_songs_folder']}/Good_Songs/Wget_Downloads/")
        except Exception:
            write_error_to_error_file(f"Error downloading song {df['song_link'][i]}")
    # # Download Bad Song Zips
    df = pd.read_csv(default_params['download_links_csv_bad'])
    for i in range(len(df)):
        try:
            wget.download(df['song_link'][i], f"{default_params['training_songs_folder']}/Bad_Songs/Wget_Downloads/")
        except Exception:
            write_error_to_error_file(f"Error downloading song {df['song_link'][i]}")


def write_error_to_error_file(error):
    with open(default_params['error_log_file'], 'a') as f:
        f.write(f"Error: {error}" + '\n')

def replace_brackets_in_folder_names():
    # Replace brackets in folder names (good)
    for root, dirs, files in os.walk(default_params['training_songs_folder'] + '/Good_Songs/'):
        for folder in dirs:
            if '[' in folder:
                os.rename(os.path.join(root, folder), os.path.join(root, folder.replace('[', '(').replace(']', ')')))
    # Replace brackets in folder names (bad)
    for root, dirs, files in os.walk(default_params['training_songs_folder'] + '/Bad_Songs/'):
        for folder in dirs:
            if '[' in folder:
                os.rename(os.path.join(root, folder), os.path.join(root, folder.replace('[', '(').replace(']', ')')))

if __name__ == "__main__":
    # scrape_pages() # Gathers data from bsaber.com
    # download_pages_to_training_songs() # Downloads songs from bsaber.com
    replace_brackets_in_folder_names() # Replaces brackets in folder names (for glob); ONLY USE AFTER EXTRACTED WITH 7ZIP