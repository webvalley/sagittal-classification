import os
import pickle
import numpy as np
import skimage.feature
import pydicom
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from skopt import gp_minimize

dirname1 = "../Data/Processed1" #npy files dir name
dirname0 = "../Data/Processed0"

"""
filenames1 = [el for el in os.listdir(dirname1) if el.endswith('.npy')]
filenames0 = [el for el in os.listdir(dirname0) if el.endswith('.npy')]
X = np.zeros((len(filenames1) + len(filenames0), 324), dtype=np.float32)
y = np.zeros(X.shape[0])
hashdict = {}
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
    hashdict[hash(X[i].tobytes())] = filename

for i, filename in enumerate(filenames1):
    img = np.load(os.path.join(dirname1, filename))[:,:, 0]
    img = skimage.filters.gaussian(img, sigma=15)
    sobel_img = skimage.filters.sobel(img)
    scaler = 10
    hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    while(hog.shape[0] != 324):
        scaler += 1
        hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/scaler), orientations=9, visualize=True, multichannel = False)
    X[i+len(filenames0)] = hog
    y[i+len(filenames0)] = 1
    hashdict[hash(X[i+len(filenames0)].tobytes())] = filename

np.save("Models/NPY/X.npy", X)
np.save("Models/NPY/y.npy", y)
"""
X = np.load("Models/NPY/X.npy")
y = np.load("Models/NPY/y.npy")

def do_stuff(inp):
    global X 
    global y
    pca = PCA(n_components=inp[0])
    pca.fit(X)
    pca_dataset = pca.transform(X)
    X_tr, X_ts, y_tr, y_ts = train_test_split(pca_dataset, y, test_size = 0.25, random_state = 42)
    rfc = RandomForestClassifier(n_estimators = 1000, max_depth = inp[1])
    svm = SVC(gamma = 'auto', probability = True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    gnb = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', lr), ('rf', rfc), ('gnb', gnb), ('svm', svm)], voting = 'soft', weights = inp[2:])
    eclf = eclf.fit(X_tr, y_tr)
    y_ts_our = eclf.predict(X_ts)

    pickle.dump(eclf, open("../Models/ML/eclf.pkl", "wb"))

    goodbad, goodgood, badgood, badbad = 0,0,0,0

    for i, ys in enumerate(y_ts):
        if(y_ts_our[i] == 1 and ys == 0):
            goodbad += 1
        elif(y_ts_our[i] == 1 and ys == 1):
            goodgood += 1
        elif(y_ts_our[i] == 0 and ys == 0):
            badbad += 1
        else:
            badgood += 1
    
    #return (-1)*fbeta_score(y_ts, y_ts_our, beta=0.5)
    print(goodgood, goodbad, badgood, badbad)
     
do_stuff([20, 20, 10, 10, 1, 10])
#res = gp_minimize(do_stuff, [(4, 32), (5,20), (1, 10), (1, 10), (1, 10), (1, 10)], verbose=True, n_calls=50)
#print(res)
#pickle.dump(res, open("gp_res.pkl", "wb"))
