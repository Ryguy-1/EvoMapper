import glob
import numpy as np
import cv2
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
# shutil
import shutil # used for deleting directories (not used right now)


# Takes Beatmap Folder and Returns numpy arrays (images) of segmented map
class Reader:

    # Params:
    # folder_directory = directory of map folder downloaded from scoresaber (full path No /)
    # difficulty_string = Easy, Normal, Hard, Expert, ExpertPlus
    # frames_per_image = frames per image group
    # time_increment = time intervals that are counted as notes -> should always be 1 for now (1 beat)
    def __init__(self, folder_directory, difficulty_string):
        self.folder_directory = folder_directory
        self.difficulty_string = difficulty_string
        self.time_steps_per_image = default_params['time_steps_per_image']
        self.time_increment = default_params['time_increment'] # something like 1 beat

        # Load Mapping Data File
        self.notes_list = self.load_notes_list()

        # Round each note to the nearest time increment
        for note in self.notes_list:
            note['_time'] = int(round(note['_time'] / self.time_increment)) * self.time_increment

        # Calculate Notes Time Range
        self.time_max = max([note['_time'] for note in self.notes_list])
    
        # Splits notes List Into Groups
        self.notes_groups = self.split_notes_list()

        # Converts Each Group to Numpy Array
        self.notes_groups_processed = self.process_notes_groups()


    # Loads List of Notes from Mapping Data File
    def load_notes_list(self):
        # Mapping Data File
        mapping_data_file = None
        dat_files = glob.glob(self.folder_directory + '/*.dat')
        
        for file in dat_files:
            if self.difficulty_string in file:
                mapping_data_file = file
                break

        # Load Mapping Data Dictionary from .dat file with pandas
        with open(mapping_data_file, 'r') as f:
            mapping_data = json.load(f)

        return mapping_data['_notes']

    # Splits Notes List Into Groups
    def split_notes_list(self):
        notes_groups = [[]]
        # Split Notes List Into Groups
        for i in np.arange(0, self.time_max, self.time_increment):
            # If most recent group has frames per image, then add new empty group to append to
            if len(notes_groups[-1]) >= self.time_steps_per_image:
                notes_groups.append([])

            # Boolean to allow finding of more than one note for each timestamp
            found = False
            # Check to see if there is a note that matches the timestamp
            for note in self.notes_list:
                if note['_time'] == i:
                    note['exists'] = True
                    
                    # If note has already been found for that time, add note to the same array
                    if found:
                        notes_groups[-1][-1].append(note)
                    else:
                        notes_groups[-1].append([note])
                        found = True
            
            # If no note matching the time stamp exists, then add a filler note with exists set to False
            if not found:
                notes_groups[-1].append([{'exists': False}])

        # If last group is smaller than max (very likely), then remove it (keep sizes constant)
        if len(notes_groups[-1])<self.time_steps_per_image:
            notes_groups = notes_groups[:-1]

        return notes_groups

    # Process Notes Groups into Numpy Arrays
    def process_notes_groups(self):
        notes_per_group = []
        for group in self.notes_groups:
            notes_per_group.append([])
            for note_array in group:
                add_note = np.zeros([3, 3, 4], dtype=np.float32)
                for note in note_array:
                    add_note = np.add(add_note, self.convert_note(note))
                notes_per_group[-1].append(add_note)

        return np.array(notes_per_group)
                
            

    # Convert Notes (Format):
    # Depth Channels: On/Off(0-1), Red/Blue/Bomb(0-3)[Excluding 2], Direction(0-8) [All Normalized]
    # Channel 0 = On/Off
    # Channel 1 = Red/Blue/Bomb
    # Channel 2 = Direction
    def convert_note(self, note_object):
        # Create Empty Slice
        empty_array = np.zeros([3, 3, 4], dtype=np.float32) # first index = depth, second index = y coord, third index = x coord

        # If note doesn't exist, return array with all zeros (including off switch on zeroith channel!)
        if note_object['exists'] == False:
            return empty_array

        # /////////////////On - Off//////////////////////////////////////
        empty_array[0][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 1 # Turns the Block On (Initialized to Zero)

        # /////////////////Red - Blue - or Bomb//////////////////////////////////////
        #    type: {0: "Left/Red", 
        #               1: "Right/Blue", 
        #               (2 is unused)
        #               3: "Bomb"} 

        if note_object['_type'] == 0: # Left/Red = 0
            empty_array[1][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 0
        elif note_object['_type'] == 1: # Right/Blue = 1
            empty_array[1][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 1
        elif note_object['_type'] == 3: # Bomb = 0.5
            empty_array[1][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 0.5

        # /////////////////Direction//////////////////////////////////////
        #     5) cut_direction: {0: "Up", 
        #                        1: "Down", 
        #                        2: "Left", 
        #                        3: "Right", 
        #                        4: "UpLeft", 
        #                        5: "UpRight", 
        #                        6: "DownLeft", 
        #                        7: "DownRight", 
        #                        8: "Any (Dot Note)"}

        if note_object['_cutDirection'] == 0:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 0/8
        elif note_object['_cutDirection'] == 1:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 1/8
        elif note_object['_cutDirection'] == 2:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 2/8
        elif note_object['_cutDirection'] == 3:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 3/8
        elif note_object['_cutDirection'] == 4:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 4/8
        elif note_object['_cutDirection'] == 5:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 5/8
        elif note_object['_cutDirection'] == 6:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 6/8
        elif note_object['_cutDirection'] == 7:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 7/8
        elif note_object['_cutDirection'] == 8:
            empty_array[2][abs(2-note_object['_lineLayer'])][note_object['_lineIndex']] = 8/8

        return empty_array

        
    def get_song_data(self):
        images = []
        for group in self.notes_groups_processed:
            image_final = None
            for image in group:
                if image_final is None:
                    image_final = image
                    continue
                # Otherwise, concatenate the images
                image_final = cv2.hconcat([image_final, image])
            images.append(image_final)
        return np.array(images)