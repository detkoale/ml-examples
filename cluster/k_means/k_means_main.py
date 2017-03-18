# coding=utf-8
from skimage.io import imread, imsave
from skimage import img_as_float
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math
import pylab

#load image as numpy array with n*m*3 size where n & m are image size. least 3 measurements holds RGB values
image = img_as_float(imread('parrots.jpg'))

#Создайте матрицу объекты-признаки: характеризуйте каждый пиксель
# тремя координатами - значениями интенсивности в пространстве RGB.
w,h,d = image.shape

def img_as_dataframe(image):
    return pd.DataFrame(np.reshape(image, (w*h,d)))

def PSNR(image1, image2):
    mse = np.mean((image1-image2)**2)
    return 10 * math.log10(float(1) / mse)

def cluster(pixels, n_clusters = 8):
    print "Clustering =",n_clusters

    pixels = pixels.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)

    means = pixels.groupby('cluster').mean().values
    mean_pixels = [means[c] for c in pixels['cluster'].values]
    mean_image = np.reshape(mean_pixels, (w, h, d))
    imsave('images/mean/parrots_' + str(n_clusters) + '.jpg', mean_image)

    medians = pixels.groupby('cluster').median().values
    median_pixels = [medians[c] for c in pixels['cluster'].values]
    median_image = np.reshape(median_pixels, (w, h, d))
    imsave('images/median/parrots_' + str(n_clusters) + '.jpg', median_image)

    return mean_image, median_image

pixels = img_as_dataframe(image)

for n in np.arange(1,21):
    mean_image, median_image = cluster(pixels, n)
    psnr_mean, psnr_median = PSNR(image, mean_image), PSNR(image, median_image)
    print psnr_mean, psnr_median

    if psnr_mean > 20 or psnr_median > 20:
        print "Answer is", n
        break



