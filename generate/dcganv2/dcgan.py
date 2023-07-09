import glob
import os
from pathlib import Path
import time
from copy import deepcopy

import numpy as np
from PIL import Image

import imageio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from utilsDataset import normImage, loadSegDataset
from utilsAugmentation import prepare_single

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu)



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Specify Constants and Hyper-Parameters
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# For loading the dataset
# datasetName = 'SWIMSEG'
# dataDir = '../datasets/swimseg/'
# datasetName = 'HYTA'
# dataDir = '../datasets/hyta/'
datasetName = 'WSISEG'
dataDir = '../../data/WSISEG-Database/'
dataType = np.float32
tfDataType = tf.float32

# Whether to perform train-test split or not
if datasetName == 'SWIMSEG' or 'WSISEG':
    mode = 'split'
elif datasetName == 'HYTA':
    mode = 'complete'
splitSeed = 41 # Random seed used for splitting the dataset
testShare = 0.2

# Hyper-parameters for DCGAN Model
BUFFER_SIZE = 256
BATCH_SIZE = 4
IMAGE_SIZE = 64

EPOCHS = 10000
noise_dim = 100
num_examples_to_generate = 16

# Results directory
resultsDir = './results/' + datasetName + str(IMAGE_SIZE) + '/'
Path(resultsDir+'logs/').mkdir(parents=True, exist_ok=True)
Path(resultsDir+'GeneratedData/').mkdir(parents=True, exist_ok=True)
Path(resultsDir+'trainImages/').mkdir(parents=True, exist_ok=True)
jsonFilePath = resultsDir + 'logs/trainingLog.json'
modelSavePath = './models/final_' + datasetName + str(IMAGE_SIZE)
ckptSavePath = './models/bestCKPT_' + datasetName + str(IMAGE_SIZE)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Empty the contents of resultsDir+'trainImages/'
files = glob.glob(resultsDir+'trainImages/*')
for f in files:
    os.remove(f)



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Read and Pre-process Data
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


images, gtMaps = loadSegDataset(dataDir, IMAGE_SIZE=IMAGE_SIZE, datasetName=datasetName, dataType=dataType)
gtMaps = np.squeeze(gtMaps, axis=3)

print(images.shape, images.dtype, np.min(images), np.max(images))
print(gtMaps.shape, gtMaps.dtype, np.min(gtMaps), np.max(gtMaps))

im = Image.fromarray(images[0].astype(np.uint8))
im.save("testIM.png")

# Extract Cloud Masks
images[:,:,:,0][gtMaps==0] = 0
images[:,:,:,1][gtMaps==0] = 0
images[:,:,:,2][gtMaps==0] = 0

im = Image.fromarray(gtMaps[0].astype(np.uint8))
im.save("testGT.png")
im = Image.fromarray(images[0].astype(np.uint8))
im.save("testCM.png")

images = normImage(images, dataType=dataType)

'''
image_filelist = glob.glob(imageDir + '*.png')
image_filelist = sorted(image_filelist)
images = np.array([np.array(Image.open(fname).resize((64,64))) for fname in image_filelist])

gtMaps_filelist = glob.glob(gtmapDir + '*.png')
gtMaps_filelist = sorted(gtMaps_filelist)
gtMaps = np.array([np.array(Image.open(fname).resize((64,64))) for fname in gtMaps_filelist])
#gtMaps = np.expand_dims(gtMaps, -1)

print(np.amin(images), np.amax(images))
print(np.amin(gtMaps), np.amax(gtMaps))
print(np.unique(gtMaps[0]))

print(images.shape, images.dtype)
print(gtMaps.shape, gtMaps.dtype)

im = Image.fromarray(images[0])
im.save("testIM.png")

images[:,:,:,0][gtMaps<127] = 0
images[:,:,:,1][gtMaps<127] = 0
images[:,:,:,2][gtMaps<127] = 0

#im = Image.fromarray(np.squeeze(gtMaps[0], axis=-1))
im = Image.fromarray(gtMaps[0])
im.save("testGT.png")
im = Image.fromarray(images[0])
im.save("testCM.png")

# Normalize the images to [-1, 1]
train_images = (images - 127.5) / 127.5
train_images = train_images.astype(np.float32)
'''

if mode == 'split':
    # Split into Train, Validation, and Test
    X_train_complete, X_test, y_train_complete, y_test = train_test_split(images, gtMaps, random_state=splitSeed, test_size=testShare)
    X_train, X_val, y_train, y_val = train_test_split(X_train_complete, y_train_complete, random_state=splitSeed, test_size=testShare)
    del(X_train_complete, y_train_complete, images, gtMaps, X_val, y_val, X_test, y_test)
else:
    X_train = deepcopy(images)
    y_train = deepcopy(gtMaps)
    del(images, gtMaps)


print(X_train.shape, X_train.dtype)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(X_train)#.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = prepare_single(train_dataset, BATCH_SIZE, shuffle=True, augment=True, shuffleBuffer=BUFFER_SIZE, dataType=tfDataType)

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Define DCGAN Model
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

generator = make_generator_model()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Define DCGAN Model Losses
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_prefix = os.path.join(ckptSavePath, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Define Training Loop
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        gen_image = (predictions[i, :, :, :].numpy()*127.5) + 127.5
        gen_image = gen_image.astype(np.uint8)
        #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.imshow(gen_image)
        plt.axis('off')

    plt.savefig(resultsDir+'trainImages/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Save the model every EPOCHS//20 epochs
        if (epoch + 1) % (EPOCHS//20) == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            # Produce images for the GIF as you go
            generate_and_save_images(generator,
                                    epoch + 1,
                                    seed)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                            epochs,
                            seed)

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(ckptSavePath))


anim_file = resultsDir + 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(resultsDir+'trainImages/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    image = imageio.v2.imread(filename)
    writer.append_data(image)


noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)

gen_image = (generated_image.numpy()*127.5) + 127.5
gen_image = gen_image.astype(np.uint8)
im = Image.fromarray(gen_image[0, :, :, :])
im.save(resultsDir + "sampleGEN.png")