def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import glob
import time
import pickle as pkl
import magic
import numpy as np
import pydicom as pd
import skimage.feature
from sklearn.svm import SVC
from sklearn.decomposition import PCA

pca = pkl.load(open("./pca.pkl", "rb"))
#model = pkl.load(open("./svm.pkl", "rb"))
model = pkl.load(open("./rfc.pkl", "rb"))

def colorcrop2(pixel_arr):
    midCol = pixel_arr.shape[1]//2
    x = pixel_arr[:,midCol:].sum(axis=0)[:,0]
    distance = np.where(x == 0)[0][0]
    ind = midCol - distance
    ind2 = midCol + distance
    #print(ind, ind2)
    return pixel_arr[:,ind:ind2]

for fn in  glob.iglob(os.path.join("/data3/wv2019/data/processed/", "*", "*", "*"), recursive=True):
    if(os.path.isdir(fn)):
        continue
    if(magic.from_file(fn) != 'DICOM medical imaging data'):
        continue
    metacrop_arr = colorcrop2(pd.read_file(fn).pixel_array)
    sobel_img = skimage.filters.sobel(metacrop_arr[:,:,0])
    scaler = 11
    hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    while(hog.shape[0] == 486):
        scaler += 1
        hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    X_red = pca.transform(np.array([hog]))
    y_pred = model.predict(X_red)
    print(y_pred, fn)
    if(y_pred[0] == 1.0):
        os.system("cp " + fn + " /data3/wv2019/data/sagittal")

