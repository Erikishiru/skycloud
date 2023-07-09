import tensorflow as tf

tf.get_logger().setLevel('ERROR')

def prepare_single(ds, batch_size, shuffle=False, augment=False, shuffleBuffer=1000, dataType=tf.float16):
    AUTOTUNE = tf.data.AUTOTUNE

    if shuffle:
        ds = ds.shuffle(shuffleBuffer)
    
    # Batch all datasets.
    ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(lambda x: (random_rotation_single(x, dataType=dataType)), 
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: (random_flip_single(x, dataType=dataType)), 
                    num_parallel_calls=AUTOTUNE)
    
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

def prepare(ds, batch_size, shuffle=False, augment=False, shuffleBuffer=1000):
    AUTOTUNE = tf.data.AUTOTUNE

    if shuffle:
        ds = ds.shuffle(shuffleBuffer)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (random_rotation(x, y)), 
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (random_flip(x, y)), 
                    num_parallel_calls=AUTOTUNE)
    
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


# Randomly rotate the input and output images
def random_rotation(input_image, output_image):
    # Generate a random angle for rotation
    angle = tf.random.uniform([], minval=0, maxval=360, dtype=tf.float32)
    
    # Rotate the input image
    input_image = tf.image.rot90(input_image, k=int(angle // 90))
    
    # Rotate the output image
    output_image = tf.image.rot90(output_image, k=int(angle // 90))
    
    return input_image, output_image

# Randomly rotate the input and output images
def random_rotation_single(input_image, dataType=tf.float32):
    # Generate a random angle for rotation
    angle = tf.random.uniform([], minval=0, maxval=360, dtype=dataType)
    
    # Rotate the input image
    input_image = tf.image.rot90(input_image, k=int(angle // 90))
    
    return input_image

# Randomly flip the input and output images
def random_flip(input_image, output_image):
    # Randomly decide whether to flip horizontally
    flip_prob = tf.random.uniform([], dtype=tf.float32)
    input_image = tf.cond(flip_prob < 0.5,
                          lambda: tf.image.flip_left_right(input_image),
                          lambda: input_image)
    output_image = tf.cond(flip_prob < 0.5,
                           lambda: tf.image.flip_left_right(output_image),
                           lambda: output_image)
    
    # Randomly decide whether to flip vertically
    flip_prob = tf.random.uniform([], dtype=tf.float32)
    input_image = tf.cond(flip_prob < 0.5,
                          lambda: tf.image.flip_up_down(input_image),
                          lambda: input_image)
    output_image = tf.cond(flip_prob < 0.5,
                           lambda: tf.image.flip_up_down(output_image),
                           lambda: output_image)
    
    return input_image, output_image

# Randomly flip the input and output images
def random_flip_single(input_image, dataType=tf.float32):
    # Randomly decide whether to flip horizontally
    flip_prob = tf.random.uniform([], dtype=dataType)
    input_image = tf.cond(flip_prob < 0.5,
                          lambda: tf.image.flip_left_right(input_image),
                          lambda: input_image)
    
    # Randomly decide whether to flip vertically
    flip_prob = tf.random.uniform([], dtype=dataType)
    input_image = tf.cond(flip_prob < 0.5,
                          lambda: tf.image.flip_up_down(input_image),
                          lambda: input_image)
    
    return input_image