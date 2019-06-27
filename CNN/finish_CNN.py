import os
import glob
import time
import magic
import numpy as np
import keras
from keras.models import load_model
import pydicom as pd
import skimage 
from ImagePreprocessing import colorcrop2
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model = load_model("../Models/Keras/model_better.h5")

for fn in glob.iglob(os.path.join("/data3/wv2019/data/processed/", "*", "*", "*"), recursive=True):
    if(os.path.isdir(fn)):
        continue
    if(magic.from_file(fn) != 'DICOM medical imaging data'):
        continue
    ds = colorcrop2(pd.read_file(fn).pixel_array)
    im_rez = skimage.transform.resize(ds, (256, 256, 1))
    output = model.predict(np.array([im_rez]))
    print(output, fn)
    if(output >= 0.9):
        os.system("cp " + fn + " /data3/wv2019/data/sagittal/9/")
    elif(output >= 0.8):
        os.system("cp " + fn + " /data3/wv2019/data/sagittal/8/")
    elif(output >= 0.7):
        os.system("cp " + fn + " /data3/wv2019/data/sagittal/7/")     
