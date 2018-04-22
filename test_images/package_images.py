import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt

labels = [1, 34, 12, 25, 18] 
images = 256*np.ones([5, 32, 32, 3])
for i in range(5):
    im = imread("%06d.jpg" % (i+1))
    print(im.shape)
    image = resize(im, (32, 32))
    images[i] = image

    plt.figure()
    plt.subplot(1,2,2)
    plt.imshow(images[i])
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.show()
