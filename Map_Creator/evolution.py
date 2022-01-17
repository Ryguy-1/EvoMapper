from hard_coded_song_from_audio import CreateHardMapping
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Map_Reader_Trainer.reader import Reader
from global_params import default_params
# From Tensorflow
from tensorflow.keras import models
# Numpy
import numpy as np



class Evolve:

    def __init__(self, data, discriminator_name = "model_0.1_1000_1000_epoch.h5"):
        self.data = data
        self.discriminator_name = discriminator_name

        # Load Discriminator
        self.discriminator = self.load_discriminator()
        
        # Evolution Loop
        self.best_per_generation = [self.data]
        self.evolution_iterations = 0
        self.copies_per_generation = 5000
        self.evolve()

    def evolve(self):
        for i in range(self.evolution_iterations):
            # Get Mutated List
            pass

    def get_mutated_list(self, map_to_mutate):
        # Get Mutated List
        mutated_list = []
        for i in range(self.copies_per_generation):
            mutated_list.append(self.mutate(map_to_mutate))
        return mutated_list

    def mutate(self, map_to_mutate):
        # Only Mutate on the frames with blocks in them
        print(map_to_mutate.shape)


    def load_discriminator(self):
        # Load Discriminator
        discriminator = models.load_model(f"{default_params['model_save_folder']}/{self.discriminator_name}.h5")
        return discriminator





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


