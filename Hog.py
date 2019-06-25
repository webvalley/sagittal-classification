import os
import skimage.feature
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

dirname = "Data/Reg" #npy files dir name

filenames = [el for el in os.listdir(dirname) if el.endswith('.npy')]
dataset = np.zeros((len(filenames), 486), dtype=np.float32)

print("Starting to Apply HOG Transform")
for i, filename in enumerate(filenames):
    img = np.load(os.path.join(dirname, filename))[:,:, 0]
    sobel_img = skimage.filters.sobel(img)
    hog, hog_image = skimage.feature.hog(skimage.transform.rescale(sobel_img, 1/11), orientations=9, visualize=True, multichannel = False)
    dataset[i] = hog

print("Starting PCA")
pca = PCA(n_components=26)
pca.fit(dataset)
pca_dataset = pca.transform(dataset)

print("Starting KMeans")
kmeans = KMeans(n_clusters=2)
kmeans.fit(pca_dataset)

import _pickle as pickle
pickle.dump(kmeans, open("model.pkl", 'wb'))

print("Starting to copy files")
for i, filename in enumerate(filenames):
    if not "rev" in filename:
        print("Label: ", kmeans.labels_[i])
        img = np.load(dirname + "/" + filename)
        ogfile = pydicom.read_file("Data/Raw/" + filename[:][:-4] + ".dcm")
        lblstr = "Label" + str(kmeans.labels_[i])
        os.system("cp Data/Raw/" + filename[:][:-4] + ".dcm " + "Data/" + lblstr)

