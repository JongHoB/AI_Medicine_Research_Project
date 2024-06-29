### 
import pandas as pd
import numpy as np
import os
import cv2

import datetime

import sys
import time
from PIL import Image

base_path='/home/csl/jhb/image_resize/'
origin_path='/home/swarm/qorwhdghjr/chexpert/chexpertchestxrays-u20210408/'


train_df=pd.read_csv(origin_path+'test_labels.csv')

def image_convert(image):
    h,w=image.shape[0:2]
    margin=[np.abs(h-w)//2,np.abs(h-w)//2]

    if np.abs(h-2)%2!=0:
        margin[0]+=1

    if h<w:
        margin_list=[margin,[0,0]]
    else:
        margin_list=[[0,0],margin]

    if len(image.shape) ==3:
        margin_list.append([0,0])

    output=np.pad(image,margin_list,mode='constant')

    return output

for i in range(len(train_df)):
    image=cv2.imread(origin_path+train_df.iloc[i,0])
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    img=Image.fromarray(image_convert(image))
    img_resize=img.resize((390,390))

    if not os.path.exists(base_path+"/".join(train_df.iloc[i,0].split('/')[0:3])):
        os.makedirs(base_path+"/".join(train_df.iloc[i,0].split('/')[0:3]))

    img_resize.save(base_path+train_df.iloc[i,0])

