import os 
import tensorflow as tf
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def detections(df, canny=True):
    """이미지 경계선 감지"""
    imgs = np.expand_dims(df, -1)
    img_data = []
    for img in imgs:
        if canny:
            median_intensity = np.median(img)
            lower_threashold = int(max(0, (1.0-0.65) * median_intensity))
            upper_threashold = int(min(255, (1.0+0.65) * median_intensity))
            img_cv = cv2.Canny(img, lower_threashold, upper_threashold)
        else:
            img_cv = cv2.Laplacian(img, -1)
        img_data.append(img_cv)
    return np.array(img_data)

def detection(df, canny=True):
    """이미지 경계선 감지"""
    if canny:
        median_intensity = np.median(df)
        lower_threashold = int(max(0, (1.0-0.65) * median_intensity))
        upper_threashold = int(min(255, (1.0+0.65) * median_intensity))
        img_cv = cv2.Canny(df, lower_threashold, upper_threashold)
    else:
        img_cv = cv2.Laplacian(df, -1)
    return img_cv

def normalize_127(image):
    img = tf.cast(np.expand_dims(image, -1), tf.float32)
    img = (img / 127.5) - 1
    return img

def normalize_255(image):
    img = tf.cast(np.expand_dims(image, -1), tf.float32)
    img /= 255.
    return img

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image
def load_image_test(input_image, real_image):
    input_image= normalize_127(input_image)
    real_image = normalize_127(real_image)
    input_image = resize(input_image,32, 32)
    real_image = resize(real_image, 32, 32)
    return input_image, real_image

def pre_test(img):
    img = normalize_127(img)
    img = resize(img , 32, 32)
    return img

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def generate_images_one(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def predict_img(model, input_img):
    """모델과 이미지를 받고
        입력이미지, 예측이미지, 예측이미지 컬러 출력"""
    # preprocess
    re_img = cv2.resize(input_img, (32,32), cv2.INTER_LANCZOS4)
    re_de = detection(re_img)
    re_de_img = normalize_127(re_de)
    
    # predict 
    pred_img = model(re_de_img[tf.newaxis, ...])
    
    # color
    rgb_img = cv2.cvtColor(np.squeeze(pred_img.numpy(), 0), cv2.COLOR_GRAY2RGB)
    
    # plot
    plt.figure(figsize=(15,15))
    display_list = [re_img, pred_img[0], rgb_img]
    title = ['Input Image', 'Predicted Image', "Color Image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def cycle_img(imgs):
    
    imgs_32 = [cv2.resize(img, (32,32), cv2.INTER_LANCZOS4) for img in imgs]
    img_de = detections(imgs_32)
    img_scale = normalize_127(img_de)   

    from codes.models import Discriminators, Generator, downsample, upsample
    OUTPUT_CHANNELS = 1

    generator_g = Generator()
    generator_f = Generator()

    discriminator_x = Discriminators()
    discriminator_y = Discriminators()

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_path = "./checkpoints_cycle"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                            generator_f=generator_f,
                            discriminator_x=discriminator_x,
                            discriminator_y=discriminator_y,
                            generator_g_optimizer=generator_g_optimizer,
                            generator_f_optimizer=generator_f_optimizer,
                            discriminator_x_optimizer=discriminator_x_optimizer,
                            discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    pred_img = generator_g(img_scale)
    pred_1 = (pred_img.numpy() + 1) * 127.5 / 255.
    return pred_1

def img_re(imgs, detection=False):
    if detection:
        imgs = detections(imgs)
    return normalize_255(imgs)

def predict_img_cycle(img_list, model, full_model):
    imgs = img_list.copy()
    imgs_32 = [cv2.resize(img, (32,32), cv2.INTER_LANCZOS4) for img in imgs]
    img_de = detections(imgs_32)
    img_scale = normalize_127(img_de)   
    pred_img = full_model(img_scale)
    pred_1 = (pred_img.numpy() + 1) * 127.5 / 255.
    dicts = {0 : "T-shirt/top",    
       1 : "Trouser",    
       2 : "Pullover",    
       3 : "Dress",    
       4 : "Coat",    
       5 : "Sandal",    
       6 : "Shirt",    
       7 : "Sneaker",    
       8 : "Bag",    
       9 : "Ankel boot"}

    imgs_28 = [cv2.resize(np.squeeze(img,-1), (28,28), cv2.INTER_LANCZOS4).reshape(28,28,1) for img in pred_1]
    pred = model.predict(np.array(imgs_28))
    label = np.argmax(pred, axis=1)

    n = len(img_list)
    fig, ax = plt.subplots(n, 2, figsize=(8, 8))
    for i, k in enumerate(img_list):
        ax[i, 0].imshow(imgs[i])
        ax[i, 1].imshow(img_list[i])
        ax[i, 0].set_title("predict : {}".format(dicts[label[i]]))
        ax[i, 0].axis("off")
        ax[i, 1].set_title("predict : {}".format(dicts[label[i]]))
        ax[i, 1].axis("off")
    fig.tight_layout()
    plt.show()