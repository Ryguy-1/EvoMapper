# Global Default Parameters
default_params = {
    'difficulty' : "ExpertPlus",
    'custom_songs_folder' : 'D:/SteamLibrary/steamapps/common/Beat Saber/Beat Saber_Data/CustomLevels',
    'data_folder' : 'Map_Reader_Trainer/.data',
    'training_songs_folder' : 'Training_Songs',
    'time_increment' : 0.1, # was 0.1
    'time_steps_per_image' : 1000, # was 100
    'train_test_split' : 0.8,
    'model_save_folder' : 'Map_Reader_Trainer/.models',
    'chromedriver_executable' : 'C:/Selenium/chromedriver.exe',
    'bsaber_good_start_website' : 'https://bsaber.com/songs/top/?time=all&difficulty=expert-plus',
    'bsaber_bad_start_website' : 'https://bsaber.com/songs/new/?difficulty=expert-plus',
    'download_links_csv_good' : 'Song_Scraper/song_links_good.csv',
    'download_links_csv_bad' : 'Song_Scraper/song_links_bad.csv',
    'error_log_file' : 'Song_Scraper/error_log.txt',
    'error_log_file_convert_songs' : 'Map_Reader_Trainer/error_log.txt',
}