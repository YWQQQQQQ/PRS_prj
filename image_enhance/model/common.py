import numpy as np
import tensorflow as tf
import cv2
DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255



def resolve_single(model, name, lr):
    if name == 'srgan':
        return resolve(model, tf.expand_dims(lr, axis=0))[0]
    elif name == 'mbllen':
        return resolve_mbllen(model, lr)

def resolve_mbllen(model, lr_batch):
    lowpercent = 5
    highpercent = 95
    maxrange = 0.8
    hsvgamma = 0.8
    b, g, r = lr_batch[:, :, 0], lr_batch[:, :, 1], lr_batch[:, :, 2]
    rgb = cv2.merge([r, g, b])
    lr_batch = tf.cast(rgb, tf.float32) / 255.
    #print(b.shape, g.shape, r.shape)

    fake_B = model.predict(tf.expand_dims(lr_batch, axis=0))[0]
    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
    percent_max = sum(sum(gray_fake_B >= maxrange)) / sum(sum(gray_fake_B <= 1.0))

    max_value = np.percentile(gray_fake_B[:], highpercent)
    if percent_max < (100 - highpercent) / 100.:
        scale = maxrange / max_value
        fake_B = fake_B * scale
        fake_B = np.minimum(fake_B, 1.0)

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
    sub_value = np.percentile(gray_fake_B[:], lowpercent)
    fake_B = (fake_B - sub_value) * (1. / (1 - sub_value))

    imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    fake_B = np.minimum(fake_B, 1.0)

    return fake_B

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


