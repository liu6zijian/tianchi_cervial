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

img_paths=glob(os.path.join("../samples","*.npz"))

for img_path in img_paths:
    print(img_path)
    a=np.load(img_path)
    roi=a['img']
    filename=img_path.split("/")[-1].split('.')[0]
    sample_path="../samples_img_png"
    save_path = os.path.join(sample_path, filename+".png")
    cv2.imwrite(save_path,roi)
    print("Finish: ",filename)

