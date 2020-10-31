from model.srgan.srgan import generator,discriminator
import model.mbllen.Network as Network

from model.common import resolve_single
from utils import cropAndResize, load_image, concatenateImage, showImage,readImage, denoise
import cv2

# Load model 1 -- Super resolution
sr_gen = generator()
sr_gen.load_weights('weight/srgan/gan_generator.h5')
# Load model 2 -- Illumination improvement

mbllen_gen = Network.build_mbllen((32, 32, 3))
mbllen_gen.load_weights('weight/mbllen/LOL_img_lowlight.h5')

# Load test image
img = readImage(r'C:\Users\ywqqq\Documents\PRS_prj\maskdetection\test.jpg')

# If noise in image, denoise.
gaussian_noise = False
salt_and_pepper_noise = False

if gaussian_noise:
    img = denoise(img, 'gaussian')

if salt_and_pepper_noise:
    img = denoise(img, 'salt-and-pepper')

# Get the luminance of the image.
# If luminance < 70, then apply illumination improvemtn
imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(imgHSV)
if V < 70:
    img = resolve_single(mbllen_gen, 'mbllen', img)

# Get the resolution of the image.
# If the resolution < 10000, then apply super resolution

r = img.shape[0]*img.shape[1]
if r < 10000:
    img = resolve_single(sr_gen, 'srgan', img)

