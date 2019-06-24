import pydicom as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import skimage.filters as filters
import skimage.feature as feature
import skimage.restoration as restoration
import skimage.exposure as exposure
import os

work_dir = './Data/Raw'
dcm_files = [os.path.join(work_dir,file_path) for file_path in os.listdir(work_dir)]

'''
Cropping by Metadata

Args: 
    img: DCM file input
Output: 
    parr: Cropped pixel array
'''
def metacrop(file):
    for key in file.dir():
        value = getattr(file, key, "")
        if(type(value) == pd.sequence.Sequence and key == "SequenceOfUltrasoundRegions"):
            value = value[0]
            break
    x0, x1, y0, y1 = None, None, None, None
    for key in value.dir():
        if key == "RegionLocationMinX0":
            x0 = getattr(value, key, "")
        if key == "RegionLocationMaxX1":
            x1 = getattr(value, key, "")
        if key == "RegionLocationMinY0":
            y0 = getattr(value, key, "")
        if key == "RegionLocationMaxY1":
            y1 = getattr(value, key, "")
    print(x0, x1, y0, y1)
    if(x0 == 0):
        return colorcrop2(file.pixel_array)
    return file.pixel_array[y0:y1,x0:x1]

'''
Cropping by Color

Args: 
    parr: Initial pixel array
Output: 
    parr: Cropped pixel array
'''
def colorcrop(pixel_arr):
    midCol = pixel_arr.shape[1]//2
    print(midCol)

    # Left Side
    flag = False
    ind = -1
    for i in range(midCol, 0, -1):
        colArr = pixel_arr[:,i,0]
        if(np.mean(colArr) == 0):
            flag = True
            ind = i
            break
    flag2 = False
    ind2 = -1
    for i in range(midCol, pixel_arr.shape[1]):
        colArr = pixel_arr[:,i,0]
        if(np.mean(colArr) == 0):
            flag2 = True
            ind2 = i
            break
    if(flag and flag2):
        return pixel_arr[:,ind:ind2]
    return pixel_arr

'''
Faster Cropping by Color

Args: 
    parr: Initial pixel array
Output: 
    parr: Cropped pixel array
'''
def colorcrop2(pixel_arr):
    midCol = pixel_arr.shape[1]//2
    x = pixel_arr[:,midCol:].sum(axis=0)[:,0]
    distance = np.where(x == 0)[0][0]
    ind = midCol - distance
    ind2 = midCol + distance
    print(ind, ind2)
    return pixel_arr[:,ind:ind2]

'''
Returns flipped image (vertically)

Args: 
    parr: Initial pixel array
Output: 
    parr: Flipped pixel array
'''
def flip(img):
    return np.fliplr(img)

'''
Gaussian Blur

Args: 
    parr: Initial pixel array
    sigma: Hyperparameter σ
Output: 
    parr: Blurred pixel array
'''
def gaussian_blur(img, sig):
    smooth_img = filters.gaussian(img, sigma = sig)
    return smooth_img

'''
Sobel Edge Detection

Args: 
    parr: Initial pixel array
Output: 
    parr: Transformed pixel array
'''
def sobel(img):
    return filters.sobel(img)

'''
Canny Edge Detection

Args: 
    parr: Initial pixel array
Output: 
    parr: Transformed pixel array
'''
def canny(img, sigma):
    return feature.canny(img, sigma = sigma)

'''
Wavelet Denoising

Args: 
    parr: Initial pixel array
Output: 
    parr: Denoised pixel array
'''
def denoising(img):
    denoise_img = restoration.denoise_wavelet(img)
    return denoise_img

'''
Contrast-Limited Adaptive Histogram Equalization

Args: 
    parr: Initial pixel array
Output: 
    parr: Transformed pixel array
'''
def clahe(img):
    return exposure.equalize_adapthist(img)

'''
Converting BW 2D array into BW 3D array

Args: 
    parr: Initial pixel array (2 dimensional)
Output: 
    parr: Transformed pixel array (3 dimensional)
'''
def conv_2d_3d(img):
    arr = np.array(np.ones((img.shape[0], img.shape[1], 3), dtype=float))
    print(arr.shape)
    arr[:,:,0] = img
    arr[:,:,1] = img
    arr[:,:,2] = img
    return arr

folder_path = "./Data/Reg"
for counter, file_path in enumerate(dcm_files):
    metacrop_arr = metacrop(pd.read_file(file_path))
    blur_arr = gaussian_blur(metacrop_arr, 15)
    clahe_arr = clahe(blur_arr)
    #canny_arr_2d = canny(clahe_arr[:,:,0], 1.5)
    #canny_arr = conv_2d_3d(canny_arr_2d)
    #denoise_arr = denoising(clahe_arr)
    #flip_edge_arr = flip(canny_arr)
    flip_arr = flip(clahe_arr)
    np.save(file_path+".npy", clahe_arr)
    np.save(file_path+"rev.npy", flip_arr)
    #np.save(file_path+"edge.npy", canny_arr)
    #np.save(file_path+"edgerev.npy", flip_edge_arr)
    
