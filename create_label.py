import numpy as np
import cv2
import os
import json
import time
import random
import pandas as pd
from glob import glob
import shutil

import cfg

img_paths=glob(os.path.join("/home/xujialang/cervical_detection/samples_img","*.jpg"))
label_paths=glob(os.path.join("../samples","*.npz"))
csv_labels = open("./csv_labels_png.csv","w")

for label_path in label_paths:
    print(label_path)
    a=np.load(label_path)
    roi=a['label']
    filename=label_path.split("/")[-1].split('.')[0]
    ab_filename=os.path.join("/home/xujialang/cervical_detection/samples_img",filename+".png")
    for i in range(roi.shape[0]):
        csv_labels.write(ab_filename+","+str(roi[i,0])+","+str(roi[i,1])+","+str(roi[i,2])+","+str(roi[i,3])+","+"pos"+"\n")
    #np.savetxt('labels.csv', my_matrix, delimiter = ',',)  
    print("Finish: ",filename)

