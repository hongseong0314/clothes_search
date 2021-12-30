import tensorflow as tf
import cv2
from tensorflow.python.client import device_lib
import numpy as np
from codes.utils import detection, detections, normalize_255

tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

# dataload
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize
x_train_scale = normalize_255(train_images)
x_val_scale = normalize_255(test_images)

# label one-hot
y_train_dum = tf.keras.utils.to_categorical(train_labels, 10)
y_val_dum = tf.keras.utils.to_categorical(test_labels, 10)

# model 정의
batchs = 128

dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_train_scale, tf.float32), 
                                              tf.cast(y_train_dum, tf.float32))).shuffle(60000).batch(batchs)

test_ds = tf.data.Dataset.from_tensor_slices((tf.cast(x_val_scale, tf.float32), 
                                             tf.cast(y_val_dum, tf.float32))).batch(batchs)

from codes.models import sai_model
model = sai_model()
opti = tf.keras.optimizers.Adam(lr = 1e-3)
loss_f = tf.keras.losses.CategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as tape:
        output = model(input_image, training=True)
        t_loss = loss_f(target, output)
    
    gradients = tape.gradient(t_loss, model.trainable_variables)
    opti.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(t_loss)
    train_accuracy(target, output)

@tf.function
def test_step(input_image, target):
    output = model(input_image, training=False)
    t_loss = loss_f(target, output)

    test_loss(t_loss)
    test_accuracy(target, output)

# 학습 시작
import time
def fit(epochs = 30):
    acc = 0
    train_acc = []
    val_acc = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        start = time.time()
        for img, label in dataset:
            print('.', end='')
            train_step(img, label)
        print()
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
            
        if acc < test_accuracy.result().numpy():
            acc = test_accuracy.result().numpy()
            model.save_weights('./check_classification/my_checkpoint')
                
        train_acc.append(train_accuracy.result().numpy())
        val_acc.append(test_accuracy.result().numpy())
        train_loss_list.append(train_loss.result().numpy())
        val_loss_list.append(test_loss.result().numpy())
            
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))
        print(f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}')

if __name__ == "__main__":
    fit(epochs=30)