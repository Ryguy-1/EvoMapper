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

# Evolving
class Evolve:

    def __init__(self, data, discriminator_name = "model_0.1_1000_1000_epoch"):
        self.data = data
        self.discriminator_name = discriminator_name

        # Load Discriminator
        self.discriminator = self.load_discriminator()
        
        # Evolution Loop
        self.best_per_generation = [self.data]
        self.evolution_iterations = 100
        self.copies_per_generation = 1000
        self.evolve()
        self.save_best_per_generation("best_per_generation_0")

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
            print(f"Best This Generation: {np.max(predictions)}")
            # Get Best (good is in index 1)
            best_index = 0
            for j in range(len(predictions)):
                if predictions[j] > predictions[best_index]:
                    best_index = j
            # Add to best_per_generation
            self.best_per_generation.append(mutated_list[best_index])
            # Print Best

    def get_mutated_list(self, map_to_mutate):
        # Get Mutated List
        mutated_list = []
        for i in range(self.copies_per_generation):
            if i%500 == 0 and i!=0:
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
                        map_to_mutate[i][0][j+k] = self.mutate_row_on_off(on_off[j+k])
                        map_to_mutate[i][1][j+k] = self.mutate_row_red_blue_bomb(red_blue_bomb[j+k], on_off_row=map_to_mutate[i][0][j+k])
                        map_to_mutate[i][2][j+k] = self.mutate_row_direction(direction[j+k], on_off_row=map_to_mutate[i][0][j+k])
        return map_to_mutate

    # Each index is either a 0 or 1 (0 = no block, 1 = block)
    def mutate_row_on_off(self, row):
        one_in_ten_list = [np.random.randint(0, 10) for i in range(len(row))]
        # if one_in_ten_list[i] == 7: then one turns to zero
        row = np.where(one_in_ten_list == 7, 0, row)
        # if one_in_ten_list[i] == 3: then zero turns to one
        row = np.where(one_in_ten_list == 3, 1, row)
        # Convert to floats
        row = row.astype(np.float32)
        return row

    # Each index is either a 0, 1, or 0.5 (0 = left/red, 1 = right/blue, 0.5 = bomb)
    def mutate_row_red_blue_bomb(self, row, on_off_row):
        # Set row to zeros where on_off_row is a zero
        row[on_off_row == 0] = 0
        # Randomly set each index in row to a 0, 0.5, or 1 if the on_off_row at that index is a 1
        row = np.where(on_off_row == 1, np.random.choice([0, 0.5, 1], size=len(row)), row)
        # Convert to floats
        row = row.astype(np.float32)
        return row

    # Each index is either a 0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, or 7/8 (0 = up, 1 = down, 2 = left, 3 = right, 4 = up/left, 5 = up/right, 6 = down/left, 7 = down/right, 8 = Any (dot note))
    def mutate_row_direction(self, row, on_off_row):
        # Set row to zeros where on_off_row is a zero
        row[on_off_row == 0] = 0
        # Randomly set each index in row to a 0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, or 7/8 if the on_off_row at that index is a 1
        row = np.where(on_off_row == 1, np.random.choice([0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8], size=len(row)), row)
        # Convert to floats
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
    # Create Mapping Using Librosa
    CreateHardMapping(song_name = song_name, song_author_name = author_name, audio_file_location = f"Map_Creator/song_temp_folder/{song_name}/song.ogg")
    # Create Reader
    reader = Reader(f"Map_Creator/song_temp_folder/Outerspacee/", default_params['difficulty'])
    # Get Song Data
    data = reader.get_song_data()
    print(data.shape)
    # Start Evolution Process using Discriminator
    evolver = Evolve(data)


