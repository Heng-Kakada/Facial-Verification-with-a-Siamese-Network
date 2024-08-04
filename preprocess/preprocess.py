import tensorflow as tf
from utils import constant

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #resize
    image = tf.image.resize(image, constant.TARGET_SHAPE)
    image = tf.cast(image, tf.float32)
    
    return image