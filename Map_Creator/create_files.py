# Imports
import os
# Contains Global Default Variables: (difficulty, save_folder)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params

class create_info_dot_dat:
    # Creates info.dat object
    # Required Parameters:
    #     song_name: Name of song
    #     song_author: Name of author
    #     bpm: Beats per minute

    def __init__(self, song_name, 
                       song_author_name,
                       bpm,
                       note_jump_movement_speed=15,
                       note_jump_start_beat_offset=0,
                       difficulty = default_params['difficulty'],
                       version = "2.0.0",
                       song_sub_name = "",
                       shuffle = 0,
                       shuffle_period = 0.5,
                       preview_start_time = 30,
                       preview_duration = 8,
                       song_filename = 'song.ogg', # May have to change this to .egg
                       cover_image_filename = 'cover.jpg',
                       environment_name = 'BigMirrorEnvironment',
                       all_directions_environment_name = "GlassDesertEnvironment",
                       song_time_offset = 0,
                       level_author_name = "BeatAutoMapper-Ryguy-1",
                       ):

        # Map Difficulty to DifficultyRank
        difficulty_rank_mappings = {
            'Easy': 1,
            'Normal': 3,
            'Hard': 5,
            'Expert': 7,
            'ExpertPlus': 9,
        }

        # Set Song Name for Save Folder
        self.song_name = song_name

        # Just Creates One Difficulty for Now
        self.info_dictionary = {
            "_version": version,
            "_songName": song_name,
            "_songSubName": song_sub_name,
            "_songAuthorName": song_author_name,
            "_levelAuthorName": level_author_name,
            "_beatsPerMinute": bpm,
            "_shuffle": shuffle,
            "_shufflePeriod": shuffle_period,
            "_previewStartTime": preview_start_time,
            "_previewDuration": preview_duration,
            "_songFilename": song_filename,
            "_coverImageFilename": cover_image_filename,
            "_environmentName": environment_name,
            "_allDirectionsEnvironmentName": all_directions_environment_name,
            "_songTimeOffset": song_time_offset,
            "_difficultyBeatmapSets": [
                {
                    "_beatmapCharacteristicName": "Standard",
                    "_difficultyBeatmaps": [
                        {
                            "_difficulty": difficulty,
                            "_difficultyRank": difficulty_rank_mappings[difficulty],
                            "_beatmapFilename": f"{difficulty}.dat",
                            "_noteJumpMovementSpeed": note_jump_movement_speed,
                            "_noteJumpStartBeatOffset": note_jump_start_beat_offset,
                        }
                    ]
                }
            ]
        }

    # Creates info.dat file locally from initialization information
    def compile_and_save(self, save_folder=default_params['custom_songs_folder']):
        # Check if Song Folder Exists Yet
        if not os.path.exists(f'{save_folder}/{self.song_name}'):
            os.makedirs(f'{save_folder}/{self.song_name}')
            print("Created Folder:", f'{save_folder}/{self.song_name}')
            
        # Create info.dat file
        with open(f'{save_folder}/{self.song_name}/info.dat', 'w') as info_dat_file:
            # Write info.dat file
            info_dat_file.write(str(self.info_dictionary))


class create_difficulty_dot_dat:

    def __init__(self, song_name, version = "2.0.0", difficulty = default_params['difficulty']):
        # General Information
        self.song_name = song_name
        self.version = version
        self.difficulty = difficulty
        
        # Object Information
        self.base_object = {
            "_version": self.version,
            "_notes": [],
            "_obstacles": [],
            "_events": [],
        }

                       
                
    # Adds Note to the notes array
    # Params:
    #     1) time: The time, in beats, where this object reaches the player.
    #     2) line_index: Column: integer number, from 0 to 3 (Left: 0 -> Right: 3)
    #     3) line_layer: Row/Layer: integer number, from 0 to 2, (Bottom: 0 -> Top: 2)
    #     4) type: {0: "Left/Red", 
    #               1: "Right/Blue", 
    #               (2 is unused)
    #               3: "Bomb"} 
    #     5) cut_direction: {0: "Up", 
    #                        1: "Down", 
    #                        2: "Left", 
    #                        3: "Right", 
    #                        4: "UpLeft", 
    #                        5: "UpRight", 
    #                        6: "DownLeft", 
    #                        7: "DownRight", 
    #                        8: "Any (Dot Note)"}
    def add_note(self, time, line_index, line_layer, type, cut_direction):
        self.base_object["_notes"].append({
            "_time": time,
            "_lineIndex": line_index,
            "_lineLayer": line_layer,
            "_type": type,
            "_cutDirection": cut_direction,
            })

    # Adds Obstacle to the obstacles array
    # Params:
    #     1) time: The time, in beats, where this object reaches the player.
    #     2) line_index: Column: integer number, from 0 to 3 (Left: 0 -> Right: 3)
    #     3) type: {0: "FullHeightWall", 
    #               1: "Crouch/DuckWall"}
    #     4) duration: The duration the obstacle extends for in beats.
    #     5) width: How many columns the obstacle takes up. (4 is full)
    def add_obstacle(self, time, line_index, type, duration, width):
        self.base_object["_obstacles"].append({
            "_time": time,
            "_lineIndex": line_index,
            "_type": type,
            "_duration": duration,
            "_width": width,
        })

    # Adds Event to the events array
    # Params:
    #     1) time: The time, in beats, where this object reaches the player.
    #     2) type: (Only utalizing values that are environment independent)
    #              {0: "Controls lights in Black Lasers group", 
    #               1: "Controls lights in Ring Lights group",
    #               2: "Controls lights in Left Rotating Lasers group",
    #               3: "Controls lights in Right Rotating Lasers group",
    #               4: "Controls lights in Center Lights group",
    #               5: "Controls boost light colors (secondary colors)",
    #               6: ENVIRONMENT DEPENDENT (Skip for now),
    #               7: ENVIRONMENT DEPENDENT (Skip for now),
    #               8: Creates one ring spin in the environment,
    #               9: Controls zoom for applicable rings. Is not affected by _value,
    #               10: ENVIRONMENT DEPENDENT (Skip for now),
    #               11: ENVIRONMENT DEPENDENT (Skip for now),
    #               12: Controls rotation speed for applicable lights in Left Rotating Lasers.
    #               13: Controls rotation speed for applicable lights in Right Rotating Lasers.
    #               14: 360/90 Early rotation. Rotates future objects, while also rotating objects at the same time.
    #               15: 360/90 Late rotation. Rotates future objects, but ignores rotating objects at the same time.
    #               16: ENVIRONMENT DEPENDENT (Skip for now),
    #               17: ENVIRONMENT DEPENDENT (Skip for now),
    #     3) value: (Controlling Lights)
    #               {0: "Turns light group off",
    #                1: "Changes the lights to blue, and turns the lights on.",
    #                2: "Changes the lights to blue, and flashes brightly before returning to normal.",
    #                3: "Changes the lights to blue, and flashes brightly before fading to black.",
    #                4: "Changes the lights to blue by fading from the current state.",
    #                5: "Changes the lights to red, and turns the lights on.",
    #                6: "Changes the lights to red, and flashes brightly before returning to normal.",
    #                7: "Changes the lights to red, and flashes brightly before fading to black.",
    #                8: "Changes the lights to blue by fading from the current state.",
    #     3) value: (Controlling Boost Colors)
    #               {0: "Turns the event off - switches to first (default) pair of colors.",
    #                1: "Turns the event on - switches to second pair of colors.",
    #     3) value: (Controlling Rings) -> When event is used to control ring zoom, the _value of event is ignored.
    #     3) value: (Controlling Cars) -> ENVIRONMENT DEPENDENT (Skip for now)
    #     3) value: (Controlling BPM) -> Skip for now
    #     3) value: (Controlling Laser Rotation Speed) -> Skip for now
    #     3) value: (Controlling 360/90 Rotation)
    #               {0: "60 Degrees Counterclockwise",
    #                1: "45 Degrees Counterclockwise",
    #                2: "30 Degrees Counterclockwise",
    #                3: "15 Degrees Counterclockwise",
    #                4: "15 Degrees Clockwise",
    #                5: "30 Degrees Clockwise",
    #                6: "45 Degrees Clockwise",
    #                7: "60 Degrees Clockwise",
    #     4) float_value: (Controlling Lights) -> _floatValue determines brightness of light.
    # }
    def add_event(self, time, type, value, float_value=0):
        self.base_object["_events"].append({
            "_time": time,
            "_type": type,
            "_value": value,
            "_floatValue": float_value,
        })

    def compile_and_save(self, save_folder = default_params["custom_songs_folder"]):
        # Check if Song Folder Exists Yet
        if not os.path.exists(f'{save_folder}/{self.song_name}'):
            os.makedirs(f'{save_folder}/{self.song_name}')
            print(f'Created Folder: {save_folder}/{self.song_name}')

        # Create difficulty.dat file
        with open(f'{save_folder}/{self.song_name}/{self.difficulty}.dat', 'w') as difficulty_dat_file:
            # Write difficulty.dat file
            difficulty_dat_file.write(str(self.base_object))