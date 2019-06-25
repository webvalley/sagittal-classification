import os
import glob
import time
import numpy as np
import keras
from keras.models import load_model
import pydicom as pd
import skimage 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model = load_model("model_better.h5")
counter = 0
for fn in glob.iglob(os.path.join("/data3/wv2019/data/processed/", "*", "*", "*"), recursive=True):
    counter += 1
    if(counter > 100):
        break
    ds = pd.read_file(fn).pixel_array
    im_rez = skimage.transform.resize(ds, (256, 256, 1))
    output = model.predict(np.array([im_rez]))
    print(output, fn)
    if(output >= 0.9):
        os.system("cp " + fn + " /data3/wv2019/data/sagittal") 
    
