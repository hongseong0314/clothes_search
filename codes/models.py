import os 
import tensorflow as tf
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result
OUTPUT_CHANNELS = 1
def Generator():
    inputs = tf.keras.layers.Input(shape=[32,32,1])
    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 16, 16, 64)
    downsample(128, 4), # (bs, 8, 8, 128)
    downsample(256, 4), # (bs, 4, 4, 256)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
      ]
    up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 512)
    upsample(512, 4), # (bs, 4, 4, 512)
    upsample(256, 4), # (bs, 8, 8, 256)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
      ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)
    x = inputs

  # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[32, 32, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[32, 32, 1], name='target_image')
    
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 32, 32, channels*2)
    
    down1 = downsample(64, 4, False)(x) # (bs, 16, 16, 64)
    down2 = downsample(128, 4)(down1) # (bs, 8, 8, 128)
    down3 = downsample(256, 4)(down2) # (bs, 4, 4, 256)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 4, 4, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 3, 3, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 5, 5, 512)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 2, 2, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def Discriminators():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[32, 32, 1], name='input_image')
    
    down1 = downsample(64, 4, False)(inp) # (bs, 16, 16, 64)
    down2 = downsample(128, 4)(down1) # (bs, 8, 8, 128)
    down3 = downsample(256, 4)(down2) # (bs, 4, 4, 256)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 4, 4, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 3, 3, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 5, 5, 512)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 2, 2, 1)

    return tf.keras.Model(inputs=[inp], outputs=last)

def Conve(filters, size, apply_dropout=False, padding="VALID"):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel_size=size, padding=padding,
                                      strides=(1,1) , kernel_initializer="he_normal"))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.25))
    result.add(tf.keras.layers.ReLU())
    return result


def sai_model(drop_out=False):
    inputs = tf.keras.layers.Input(shape=[28,28,1])
    
    output = Conve(64,3)(inputs) #26 x 26 x 64    
    output = Conve(64,3)(output) #24 x 24 x 64
    
    inception_1 = tf.keras.layers.MaxPool2D(pool_size=2)(output)
    inception_1 = Conve(128, 3, padding="SAME")(inception_1)
    
    inception_2 = Conve(128, 5, padding="VALID")(output)
    inception_2 = Conve(128, 5, padding="VALID")(inception_2)
    inception_2 = Conve(128, 5, padding="VALID")(inception_2)
    
    output = tf.keras.layers.concatenate([inception_1, inception_2])
    
    inception_1 = tf.keras.layers.MaxPool2D(pool_size=2)(output)
    inception_1 = Conve(256, 3, padding="VALID")(inception_1)
    
    inception_2 = Conve(256, 5, padding="VALID")(output)
    inception_2 = Conve(256, 3, padding="VALID")(inception_2)
    inception_2 = Conve(256, 3, padding="VALID")(inception_2)
    
    output = tf.keras.layers.concatenate([inception_1, inception_2])
    output = Conve(512, 2, padding="SAME")(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(512, activation="relu")(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(10, activation="softmax")(output)
    return tf.keras.Model(inputs=inputs, outputs=output)
