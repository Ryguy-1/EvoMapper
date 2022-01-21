from hard_coded_song_from_audio import CreateHardMapping
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Map_Reader_Trainer.reader import Reader
from global_params import default_params
# From Tensorflow
from tensorflow.keras import models
# Numpy
import numpy as np
# Print Group Pretty
from Map_Reader_Trainer.main import print_group_pretty
# Pickle
import pickle
# Data to Mapping
from data_to_mapping import Data_To_Mapping

# Evolving
class Evolve:

    def __init__(self, data, discriminator_name = "model1_0.1_1000_15_epoch", save_name = "best_per_generation"):
        self.data = data
        self.discriminator_name = discriminator_name
        self.save_name = save_name

        # Print Information
        self.print_interval = 25
        self.print_end_rankings_array = False

        # Load Discriminator
        self.discriminator = self.load_discriminator()
        
        # Prob Denominator (1/prob_mutate_index) = percent chance of mutating each index
        self.prob_mutate_index = 500

        # Evolution Loop
        self.best_per_generation = [self.data]
        self.evolution_iterations = 100
        self.copies_per_generation = 200
        self.early_stop_percentage = 0.24
        self.evolve()
        self.save_best_per_generation(self.save_name)

    def evolve(self):
        for i in range(self.evolution_iterations):
            # Get Mutated List
            mutated_list = self.get_mutated_list(self.best_per_generation[-1])
            predictions = []
            # Get Discriminator Predictions
            for song in mutated_list:
                # For each song take each section and get the prediction -> Average the predictions to get a prediction for the overall song
                this_prediction = self.discriminator.predict(song)
                positive = 0; negative = 0
                for value_arr in this_prediction:
                    positive += value_arr[1]
                    negative += value_arr[0]
                percent_good = positive/(positive + negative)
                predictions.append(percent_good)
            # Printout
            if self.print_end_rankings_array:
                print(f"Rankings: {predictions}")
            print(f"Best This Generation: {np.max(predictions)}")
            
            # Get Best Song
            best_song = mutated_list[np.argmax(predictions)]
            # Add to best_per_generation
            self.best_per_generation.append(best_song)

            if np.max(predictions) > self.early_stop_percentage:
                return

    def get_mutated_list(self, map_to_mutate):
        # Get Mutated List
        mutated_list = []
        # Add copy of map_to_mutate to mutated_list (if doesn't get better (unlikely), then next generation will be from original copy)
        mutated_list.append(map_to_mutate.copy())
        for i in range(self.copies_per_generation):
            if i%self.print_interval == 0 and i!=0:
                print(f"Mutated {i} / {self.copies_per_generation} in generation {len(self.best_per_generation)}")
            mutated_list.append(self.mutate(map_to_mutate.copy()))
        return mutated_list

    def mutate(self, map_to_mutate):
        # Only Mutate on the frames with blocks in them
        # Iterate over each image
        for i in range(map_to_mutate.shape[0]):
            training_frame = map_to_mutate[i]
            (channels, y, x) = training_frame.shape
            on_off = training_frame[0]
            red_blue_bomb = training_frame[1]
            direction = training_frame[2]

            for j in range(0, y, 3):

                # If either on_off[i], on_off[i+1], or on_off[i+2] contain something other than zeros, then mutate
                if np.count_nonzero(on_off[j]) > 0 or np.count_nonzero(on_off[j+1]) > 0 or np.count_nonzero(on_off[j+2]) > 0:
                    # Mutate
                    for k in range(0, 3):
                        # Random number between 0 and prob_mutate_index
                        rand_num = np.random.randint(0, self.prob_mutate_index)
                        # If rand_num = 42, then mutate an index in the row
                        if rand_num == 42:
                            # Rand_Index
                            rand_index = np.random.randint(0, 4)
                            # Mutate on_off
                            map_to_mutate[i][0][j+k] = self.mutate_index_in_row_on_off(on_off[j+k], rand_index = rand_index)
                            # Mutate red_blue_bomb
                            map_to_mutate[i][1][j+k] = self.mutate_index_in_row_red_blue_bomb(red_blue_bomb[j+k], on_off_row = map_to_mutate[i][0][j+k], rand_index = rand_index)
                            # Mutate direction
                            map_to_mutate[i][2][j+k] = self.mutate_index_in_row_direction(direction[j+k], on_off_row = map_to_mutate[i][0][j+k], rand_index = rand_index)
                        # map_to_mutate[i][0][j+k] = self.mutate_row_on_off(on_off[j+k])
                        # map_to_mutate[i][1][j+k] = self.mutate_row_red_blue_bomb(red_blue_bomb[j+k], on_off_row=map_to_mutate[i][0][j+k])
                        # map_to_mutate[i][2][j+k] = self.mutate_row_direction(direction[j+k], on_off_row=map_to_mutate[i][0][j+k])
        return map_to_mutate

    # WILL IMPLEMENT IF IT TURNS OUT THERE ARE LITERALLY NO NOTES OR NOTES AREN'T ON THE IMPORTANT BEATS, ETC. -> FORCES NOTES TO BE ON EVERY BEAT (SWAP IN FOR LINE 89) (call at beginning and use in 89)
    # Get values of y (rows per image) with pixels that are on in that row, row+1, or row+2 and return array of shape = [map_to_mutate.shape[0], indices_of_y_values_with_pixels_on_in_that_row_or_row+1_or_row+2]
    def get_y_values_with_pixels_on(self, map_to_mutate):
        # Y Values with Pixels on
        y_values_with_pixels_on = [[] for i in range(map_to_mutate.shape[0])]
        # For Each Training Frame
        for i in range(map_to_mutate.shape[0]):
            training_frame = map_to_mutate[i]
            (channels, y, x) = training_frame.shape
            on_off = training_frame[0]
            # For Each Map Frame
            for j in range(0, y, 3):
                # If either on_off[i], on_off[i+1], or on_off[i+2] contain something other than zeros, then mutate
                if np.count_nonzero(on_off[j]) > 0 or np.count_nonzero(on_off[j+1]) > 0 or np.count_nonzero(on_off[j+2]) > 0:
                    y_values_with_pixels_on[i].append(j)
        return y_values_with_pixels_on

    def mutate_index_in_row_on_off(self, row, rand_index):
        # Mutate Index
        row[rand_index] = np.random.choice([0, 1], p=[0.5, 0.5])
        # Convert to float
        row = row.astype(np.float32)
        return row

    def mutate_index_in_row_red_blue_bomb(self, row, on_off_row, rand_index):
        # Set row to zeros where on_off_row is a zero
        row[on_off_row == 0] = 0
        # Mutate Index in Row
        row[rand_index] = np.random.choice([0, 0.5, 1], p=[0.5, 0, 0.5])
        # Set to Floats
        row = row.astype(np.float32)
        return row

    def mutate_index_in_row_direction(self, row, on_off_row, rand_index):
        # Set row to zeros where on_off_row is a zero
        row[on_off_row == 0] = 0
        # Mutate Index in Row
        row[rand_index] =  np.random.choice([0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1], p=[1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9])
        # Set to Floats
        row = row.astype(np.float32)
        return row

    # Discriminator Model
    def load_discriminator(self):
        # Load Discriminator
        discriminator = models.load_model(f"{default_params['model_save_folder']}/{self.discriminator_name}.h5")
        return discriminator

    # Pickle Save
    def save_best_per_generation(self, name):
        # Save Best Per Generation
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(self.best_per_generation, f)

# Main
if __name__ == "__main__":
    song_name = "Outerspacee"
    author_name = "Your's truly"
    audio_file_location = f"Map_Creator/song_temp_folder/{song_name}/song.ogg"
    # Create Mapping Using Librosa
    CreateHardMapping(song_name = song_name, song_author_name = author_name, audio_file_location = audio_file_location)
    # Create Reader
    reader = Reader(f"Map_Creator/song_temp_folder/Outerspacee/", default_params['difficulty'])
    # Get Song Data
    data = reader.get_song_data()
    print(data.shape)
    # Start Evolution Process using Discriminator
    evolver = Evolve(data, discriminator_name = "model1_0.1_1000_100_epoch", save_name="best_per_generation_1")


