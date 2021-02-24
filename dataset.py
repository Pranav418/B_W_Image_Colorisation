from google.colab import drive
drive.mount('/content/drive')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from google.colab import files

from google.colab import drive
from google.colab.patches import cv2_imshow

path='/content/drive/MyDrive/lab/tfrecords/'          # path must point to the tfrecords

path2='/content/drive/MyDrive/lab/'

filenames = os.listdir(path)
filenames = [path+x for x in filenames]

import tensorflow as tf

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,Reshape,Conv2D,Dropout,multiply,Dot,Concatenate,subtract,ZeroPadding2D,UpSampling2D
from tensorflow.keras.layers import BatchNormalization,LeakyReLU,Flatten
from tensorflow.keras.layers import Conv2DTranspose as Deconv2d
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, optimizers
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import time

def get_pair(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string)


        }
    
    example = tf.io.parse_single_example(example, tfrec_format)
    inp,lab=decode_image(example)
        
    x = (inp,lab)
    
    return x

def decode_image(example):
    
    img=example['image']
    
    
    
    img = tf.image.decode_jpeg(img, channels=3)
    l_img=tf.expand_dims(img[:,:,0],axis=2)
    a_img=tf.expand_dims(img[:,:,1],axis=2)
    b_img=tf.expand_dims(img[:,:,2],axis=2)
    l_img = (tf.cast(l_img, tf.float32)/127.5)-1
    a_img= ((tf.cast(a_img, tf.float32)-34)/105.5)-1
    b_img= ((tf.cast(b_img, tf.float32)-2)/120)-1
    lab_img=tf.concat([l_img,a_img,b_img],axis=2)
    # l_img = (tf.cast(l_img, tf.float32) / 127.5)-1
    # l_img=l_img/255
    # l_img = tf.image.resize(l_img, (256, 256), method='nearest')
    # l_img=l_img[:,:,1:]

    # in_img,l_img= augment(in_img,l_img)
    
    # in_img=tf.image.rgb_to_grayscale(in_img)
   
    return l_img,lab_img
                                                                                ##### MAKE INTO FUNCTION
train_data = tf.data.TFRecordDataset( 
    filenames,
    num_parallel_reads = tf.data.experimental.AUTOTUNE
)

ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False 
train_data = train_data.with_options(ignore_order)

train_data = train_data.map(
    get_pair, 
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

train_data = train_data.batch(8).shuffle(10)

x_shape,y_shape=(256,256,1),(256,256,3)






