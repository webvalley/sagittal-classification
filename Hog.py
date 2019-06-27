import os
import pickle
import skimage.feature
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split

dirname1 = "Data/Processed1" #npy files dir name
dirname0 = "Data/Processed0"

filenames1 = [el for el in os.listdir(dirname1) if el.endswith('.npy')]
filenames0 = [el for el in os.listdir(dirname0) if el.endswith('.npy')]
X = np.zeros((len(filenames1) + len(filenames0), 324), dtype=np.float32)
y = np.zeros(X.shape[0])
print("Starting to Apply HOG Transform")
hog, hog_image = None, None
for i, filename in enumerate(filenames0):
    img = np.load(os.path.join(dirname0, filename))[:,:,0]
    img = skimage.filters.gaussian(img, sigma=15)
    sobel_img = skimage.filters.sobel(img)
    scaler = 10
    hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    while(hog.shape[0] != 324):
        scaler += 1
        hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    X[i] = hog
    y[i] = 0

for i, filename in enumerate(filenames1):
    img = np.load(os.path.join(dirname1, filename))[:,:, 0]
    img = skimage.filters.gaussian(img, sigma=15)
    sobel_img = skimage.filters.sobel(img)
    scaler = 10
    hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    while(hog.shape[0] != 324):
        scaler += 1
        hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    X[i] = hog
    y[i] = 1

np.save("Models/NPY/X.npy", X)
np.save("Models/NPY/y.npy", y)

X = np.load("Models/NPY/X.npy")
y = np.load("Models/NPY/y.npy")

pca = PCA(n_components=16)
pca.fit(X)
pca_dataset = pca.transform(X)

X_tr, X_ts, y_tr, y_ts = train_test_split(pca_dataset, y, test_size = 0.25, random_state = 42)

rfc = RFC(n_estimators = 1000, max_depth = 10)
rfc.fit(X_tr, y_tr)
y_ts_our_rfc = rfc.predict(X_ts)

svm = SVC(gamma = 'auto')
svm.fit(X_tr, y_tr)
y_ts_our_svm = svm.predict(X_ts)

pickle.dump(svm, open("svm.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))
pickle.dump(rfc, open("rfc.pkl", "wb"))
tresh = 0.5
cor1 = 0
tot1 = 0
for i, y in enumerate(y_ts):
    if(y == 1 and y_ts_our_svm[i] > tresh):
        cor1 += 1
    if(y == 0 and y_ts_our_svm[i] <= tresh):
        cor1 += 1
    tot1 += 1
print(cor1/tot1)

cor2 = 0
tot2 = 0
for i, y in enumerate(y_ts):
    if(y == 1 and y_ts_our_rfc[i] > tresh):
        cor2 += 1
    if(y == 0 and y_ts_our_rfc[i] <= tresh):
        cor2 += 1
    tot2 += 1
print(cor2/tot2)
pos = 0
neg = 0
good = 0
bad = 0
for i, y in enumerate(y_ts):
    if(y == 1):
        pos += 1
    else:
        neg += 1
    if(y_ts_our_rfc[i] == 1 and y_ts_our_svm[i] == 1 and y == 0):
        bad += 1
    elif(y_ts_our_rfc[i] == 1 and y_ts_our_svm[i] == 1):
        good += 1
print(pos, neg, good, bad)
    
