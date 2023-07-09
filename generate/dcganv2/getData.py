import tensorflow as tf
from tensorflow.keras import layers

print(tf.__version__)
tf.get_logger().setLevel('ERROR')

import glob
import numpy as np
import os
from PIL import Image, ImageFilter
from copy import deepcopy


# For identifying the model
datasetName = 'SWIMSEG'
# datasetName = 'HYTA'
IMAGE_SIZE = 64
dataType = np.float32
tfDataType = tf.float32

num_gen_images = 300

noise_dim = 100
resultsDir = './results/' + datasetName + str(IMAGE_SIZE) + '/'
ckptSavePath_sky = './models/bestCKPTsky' + str(IMAGE_SIZE)
ckptSavePath_cmask = './models/bestCKPT_' + datasetName + str(IMAGE_SIZE)

# Empty the contents of resultsDir+'trainImages/'
files = glob.glob(resultsDir+"GeneratedData/*")
for f in files:
    os.remove(f)



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



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Restore Training Checkpoints
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

generator_sky = make_generator_model()
discriminator_sky = make_discriminator_model()

checkpoint_sky_prefix = os.path.join(ckptSavePath_sky, "ckpt")
checkpoint_sky = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_sky,
                                 discriminator=discriminator_sky)
checkpoint_sky.restore(tf.train.latest_checkpoint(ckptSavePath_sky))

generator_cmask = make_generator_model()
discriminator_cmask = make_discriminator_model()
checkpoint_cmask_prefix = os.path.join(ckptSavePath_cmask, "ckpt")
checkpoint_cmask = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_cmask,
                                 discriminator=discriminator_cmask)
checkpoint_cmask.restore(tf.train.latest_checkpoint(ckptSavePath_cmask))



#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Generate and Save Images
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# Define 'thr' value below which the pixel in cloudMask should be considered as sky
thr = 50
# Define 'midThr' value above which the pixel in cloudMask should be considered as cloud but below which main image has some cloud bondaries
midThr = 120
# Define 'upperThr' value above which the pixel in cloudMask should be considered as cloud
upperThr = 150

def smoothImage(image):
    imgCopy = image.copy()
    windowSize = 5 # Should be an odd number
    for i in range(imgCopy.shape[0]):
        for j in range(imgCopy.shape[1]):
            rowStartIdx = int(np.max([0, i-(windowSize-1)/2]))
            rowEndIdx = int(np.min([imgCopy.shape[0]-1, i+(windowSize-1)/2]))
            colStartIdx = int(np.max([0, j-(windowSize-1)/2]))
            colEndIdx = int(np.min([imgCopy.shape[1]-1, j+(windowSize-1)/2]))
            temp = image[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx].flatten()
            imgCopy[i, j] = int(np.bincount(temp).argmax())
    imgCopy = imgCopy.astype('uint8')
    return imgCopy

i = 0
while i < num_gen_images:
    noise = tf.random.normal([1, noise_dim])
    generated_sky = generator_sky(noise, training=False)
    generated_sky = (generated_sky.numpy()*127.5) + 127.5
    generated_sky = generated_sky.astype(np.uint8)
    skyImage = np.array(Image.fromarray(generated_sky[0, :, :, :]))

    generated_cmask = generator_cmask(noise, training=False)
    generated_cmask = (generated_cmask.numpy()*127.5) + 127.5
    generated_cmask = generated_cmask.astype(np.uint8)
    cloudMask = np.array(Image.fromarray(generated_cmask[0, :, :, :]))

    #skyImage = np.array(Image.open('sampleGEN_sky.png'))
    #cloudMask = np.array(Image.open('sampleGEN.png'))#.filter(ImageFilter.GaussianBlur(radius = 1)))

    print(i, skyImage.shape, skyImage.dtype)
    print(i, cloudMask.shape, cloudMask.dtype)

    genImg = deepcopy(cloudMask)
    genGTmap = np.zeros(cloudMask.shape)

    for row in range(cloudMask.shape[0]):
        for col in range(cloudMask.shape[1]):
            if cloudMask[row,col,0]<thr and cloudMask[row,col,1]<thr and cloudMask[row,col,2]<thr:
                genImg[row,col,:] = skyImage[row,col,:]
            elif cloudMask[row,col,0]<upperThr and cloudMask[row,col,1]<upperThr and cloudMask[row,col,2]<upperThr:
                genImg[row,col,0] = (genImg[row,col,0] * ((genImg[row,col,0]-thr)/(upperThr-thr))) + \
                                    (skyImage[row,col,0] * (1.0 -((genImg[row,col,0]-thr)/(upperThr-thr))))
                genImg[row,col,1] = (genImg[row,col,1] * ((genImg[row,col,1]-thr)/(upperThr-thr))) + \
                                    (skyImage[row,col,1] * (1.0 - ((genImg[row,col,1]-thr)/(upperThr-thr))))
                genImg[row,col,2] = (genImg[row,col,2] * ((genImg[row,col,2]-thr)/(upperThr-thr))) + \
                                    (skyImage[row,col,2] * (1.0 - ((genImg[row,col,2]-thr)/(upperThr-thr))))
                if cloudMask[row,col,0]>midThr and cloudMask[row,col,1]>midThr and cloudMask[row,col,2]>midThr:
                    genGTmap[row,col,:] = 255
            else:
                genGTmap[row,col,:] = 255
    
    resizeDim = 256

    genGTmap = genGTmap.astype(np.uint8)
    smooth_genGTmap = smoothImage(genGTmap)
    if np.sum(smooth_genGTmap) < 5:
        print("Regenerating due to black GTmap!")
        continue

    im = Image.fromarray(smooth_genGTmap)#.resize((resizeDim, resizeDim))
    im = im.convert("L")
    #im = np.array(Image.fromarray(genGTmap).filter(ImageFilter.GaussianBlur(radius = 1)))
    im.save(resultsDir+"GeneratedData/"+str(i)+"_genGTmap.png")

    skyImage = skyImage.astype(np.uint8)
    im = Image.fromarray(skyImage)#.resize((resizeDim, resizeDim))
    im.save(resultsDir+"GeneratedData/"+str(i)+"_genSkyImage.png")

    cloudMask = cloudMask.astype(np.uint8)
    im = Image.fromarray(cloudMask)#.resize((resizeDim, resizeDim))
    im.save(resultsDir+"GeneratedData/"+str(i)+"_genMaskImage.png")

    genImg = genImg.astype(np.uint8)
    im = Image.fromarray(genImg).filter(ImageFilter.GaussianBlur(radius = 1))#.resize((resizeDim, resizeDim))
    im.save(resultsDir+"GeneratedData/"+str(i)+"_genImage.png")

    i+=1