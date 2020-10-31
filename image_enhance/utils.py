import matplotlib.pyplot as plt
from PIL import Image
import copy
import tensorflow as tf
from keras import backend as K
import numpy as np
import cv2 as cv

def bright_mae(y_true, y_pred):
    return K.mean(K.abs(y_pred[:,:,:,:3] - y_true[:,:,:,:3]))

def bright_mse(y_true, y_pred):
    return K.mean((y_pred[:,:,:,:3] - y_true[:,:,:,:3])**2)

def bright_AB(y_true, y_pred):
            return K.abs(K.mean(y_true[:,:,:,:3])-K.mean(y_pred[:,:,:,:3]))

def log10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def bright_psnr(y_true, y_pred):
    mse = K.mean((K.abs(y_pred[:,:,:,:3] - y_true[:,:,:,:3])) ** 2)
    max_num = 1.0
    psnr = 10 * log10(max_num ** 2 / mse)
    return psnr

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                            (sigma1_sq + sigma2_sq + C2)),
                        (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                            (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                    (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def bright_SSIM(y_true, y_pred):
    SSIM_loss = tf_ssim(tf.expand_dims(y_pred[:,:,:,0], -1), tf.expand_dims(y_true[:,:,:,0], -1))+tf_ssim(tf.expand_dims(y_pred[:,:,:,1], -1), tf.expand_dims(y_true[:,:,:,1], -1)) + tf_ssim(tf.expand_dims(y_pred[:,:,:,2], -1), tf.expand_dims(y_true[:,:,:,2], -1))
    return SSIM_loss/3

def psnr_cau(y_true, y_pred):
    mse = np.mean((np.abs(y_pred - y_true)) ** 2)
    max_num = 1.0
    psnr = 10 * np.log10(max_num ** 2 / mse)
    return psnr

def save_model(model, name, epoch, batch_i):
    modelname = './Res_models/' + str(epoch) + '_' + str(batch_i) + name + '.h5'
    model.save_weights(modelname)

def imread_color(path):
    img = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH) / 255.
    b, g, r = cv.split(img)
    print(b.shape, g.shape,r.shape)
    img_rgb = cv.merge([r, g, b])
    return img_rgb
    # return scipy.misc.imread(path, mode='RGB').astype(np.float) / 255.

def imwrite(path, img):
    r, g, b = cv.split(img*255)
    img_rgb = cv.merge([b, g, r])
    cv.imwrite(path, img_rgb)
    # scipy.misc.toimage(img * 255, high=255, low=0, cmin=0, cmax=255).save(path)

def range_scale(x):
    return x * 2 - 1.

def readImage(path, color_or_gray='color'):
    if color_or_gray == 'color':
        color = 1
    elif color_or_gray == 'gray':
        color = 0
    else:
        raise NameError('Parameter color_or_gray can only be color or gray')

    img = cv.imread(path, color)
    if img is None:
        raise ValueError('Please check your image path!')
    return img


def showImage(img, wait_key=0):
    cv.imshow('image', img)
    cv.waitKey(wait_key)
    cv.destroyAllWindows()





def gammaAdjustment(img, gamma=1.0):
    table = []
    tmp_img = copy.deepcopy(img)
    for i in range(256):
        table.append(((i / 255.0) ** gamma) * 255)
    table = np.array(table).astype("uint8")
    return cv.LUT(tmp_img, table)


def cropAndResize(img, bounding_box=None, new_size=None):
    if len(img.shape) == 3:
        padding_value = (0, 0, 0)
    elif len(img.shape) == 2:
        padding_value = (0)
    else:
        raise ValueError('Unknown image format')

    if bounding_box is None:
        x1, y1 = 0, 0
        x2, y2 = img.shape[1], img.shape[0]
    else:
        x1, y1 = bounding_box[0]
        x2, y2 = bounding_box[1]

    if new_size is None:
        w, h = x2 - x1, y2 - y1
    else:
        w, h = x2 - x1, y2 - y1
        ratio = min(new_size[0] / h, new_size[1] / w)
        w, h = int(np.round(ratio * w)), int(np.round(ratio * h))

    cropped_img = img[y1:y2, x1:x2]
    resized_img = cv.resize(cropped_img, (h, w))

    top = bottom = (new_size[0] - h) // 2
    left = right = (new_size[1] - w) // 2

    padded_image = cv.copyMakeBorder(resized_img, top, bottom, left, right, cv.BORDER_CONSTANT, value=padding_value)
    return padded_image








def concatenateImage(img1, img2):
    concatenated_img = np.concatenate((img1, img2), axis=1)
    return concatenated_img



def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def sharpen(img):
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    tmp_img = copy.deepcopy(img)
    sharpened_img = cv.filter2D(tmp_img, -1, sharpen_filter)

    return sharpened_img

def denoise(img, type='gaussian'):
    if type == 'gaussian':
        denoised_img = cv.blur(img, 3)
    elif type == 'salt-and-pepper':
        denoised_img = cv.medianBlur(img, (3,3))
    else:
        raise ValueError('Unknown noise')
    return denoised_img

def equalizeHist(img, clipLimit=2.0, tileGridSize=(8, 8)):
    tmp_img = copy.deepcopy(img)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    if len(img.shape) == 3:
        shape = img.shape
        for c in range(shape[2]):
            tmp_img[:, :, c] = clahe.apply(img[:, :, c])

    elif len(img.shape) == 2:
        tmp_img = clahe.apply(img)

    else:
        raise ValueError('Unknown image format')

    return tmp_img

