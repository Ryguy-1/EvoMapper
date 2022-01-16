import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
from IPython import display
import PIL



# Load Numpy Arrays using the tensorflow dataset helper method
(train_images, train_labels), (_, _) = mnist.load_data()

# Print out the shapes of each numpy array
print(train_images.shape)
print(train_labels.shape)

# Reshape the numpy arrays into a channels last format
train_images = train_images.reshape(-1, 28, 28, 1) # Channels last format
# Make sure all of the datatypes are float32
train_images = train_images.astype('float32') # convert to float32
# Normalize each value in the numpy array dataset to between -1 and 1
train_images = (train_images-127.5)/127.5

# Set the batch size
# Hyperparameters
batch_size = 32

# Convert the training images into a tensorflow dataset -> Very cool method -> creates an iterator also!!
dataset = tf.data.Dataset.from_tensor_slices(train_images)
# Shuffle the dataset (have to pass in length for some reason)
dataset = dataset.shuffle(len(train_images))
# Set the batch size when you iterate over it ahead of time
dataset = dataset.batch(batch_size)


# Generator model -> Takes 'random' noise and creates arbitrary images out of them that should in theory be indistinguishible from original training set
# I believe you could also feed this some stimulus and have its output be linked to it. (song -> beat mapping)
# Song in very large audio spectrogram and beat map be encoded into the pixels of an image (image to image!)

# For building one from scratch, I've gathered that you just multiply the input size by strides to get the output size with set channels
# Padding = 'same' may be important for this I hypothesise!
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Return generator model object
generator = make_generator_model()

# Test with random noise to ensure working
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Display random generated image pre-train
plt.imshow(generated_image[0, :, :, :], cmap='gray')
plt.show()


# Discriminator model takes image of same output size as generator and returns 1 for real and -1 for fake
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Create object of discriminator model
discriminator = make_discriminator_model()
# Print out decision for image generated with random stimulus
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss -> BinaryCrossentropy because its either real or fake!
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# This function is just constant I believe and dependent on the binary loss function above ^
# Come back to understand this in far future
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Come back to understand this in far future
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Generator and discriminator both use adam here because adam pogu
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Just saves images of each output at certain points -> May leave out if take too much time
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Epochs, 
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

# Will take in audio spectrogram batch, yes, but also take in actual beat maps
@tf.function
def train_step(images):
    # Creates one dimensional list/array of 100 noise points
    # Cut this part out and swap for real spectragram
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # Gets batch of generated images -> instead of noise, pass in audio spectragram
      generated_images = generator(noise, training=True)
      # Gets the discriminator's output for the real image batch -> instead of images, pass in corralated beat mappings (actual)
      real_output = discriminator(images, training=True)
      # Gets the discriminator's output for the generated image batch -> instead of images, pass in generated beat mappings (predicted)
      fake_output = discriminator(generated_images, training=True)

      # Calculates the generator's loss with only its own output
      gen_loss = generator_loss(fake_output)
      # Calculates the discriminator's correctness/loss
      disc_loss = discriminator_loss(real_output, fake_output)

    # Get gradients and update with optim
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Train on dataset
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      # Will also have to pass in corrolated beat mappings
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()



train(dataset, EPOCHS)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)