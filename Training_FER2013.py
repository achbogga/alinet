
# coding: utf-8

# In[1]:

import numpy as np
import scipy.misc


# In[2]:

import cv2
from sklearn.cross_validation import train_test_split


# In[3]:

from keras.utils import np_utils
import keras.backend as K


# In[4]:

with open('fer2013/fer2013.csv') as txt:
    rows = txt.read().splitlines()


# In[5]:

fer_data = np.loadtxt((x.replace(b',',b' ') for x in rows))


# In[6]:

print fer_data.shape


# In[9]:

fer_images = fer_data[:,1:2305]
fer_images = fer_images.reshape((35887, 48, 48))
fer_labels = imageLabels = fer_data[:,0]
fer_train_test_given = fer_data[:,2305]


# In[10]:

print fer_images.shape
imageData = np.array(fer_images, dtype='uint8')
img_rows = img_cols = 48


# In[43]:

'''
scale_factor = 1.1
min_neighbors = 3
min_size = (20, 20)
#flags = cv2.CV_HAAR_SCALE_IMAGE
img_rows = img_cols = 48

imageData = np.array(fer_images, dtype='uint8')
face_detection_xml ="/home/achbogga/opencv2_data/haarcascades/haarcascade_frontalface_default.xml"
faceDetectClassifier = cv2.CascadeClassifier(face_detection_xml)

imageDataFin = []
index = 0
non_pr = 0
for i in imageData:
    #print index
    index += 1
    facePoints = faceDetectClassifier.detectMultiScale(i,  scaleFactor = scale_factor, minNeighbors = min_neighbors, minSize = min_size)
    if (len(facePoints) > 0):
        x,y,w,h = facePoints[0]
        cropped = i[y:y+h, x:x+w]
        face = np.resize(cropped, (img_rows,img_cols))
    else:
        #print "face is not detected by opencv in sample: ", index, "\n"
        face = i
        non_pr+=1
    imageDataFin.append(face)
print "faces not detected by opencv in: ",non_pr," samples"
''' and None


# In[48]:

out_put_classes = 7
img_chs = 3
def copy_to_n_channels(input_arr, n):
    temp = np.zeros(input_arr.shape+(n,), dtype='float32')
    for i in range(n):
        temp[:, :, :, i] = input_arr
    return temp

X_train, X_test, y_train, y_test = train_test_split(np.array(imageData),np.array(imageLabels), train_size=0.9, random_state = 7)
X_train = np.array(X_train)
X_test = np.array(X_test)

nb_classes = out_put_classes
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[14]:

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = copy_to_n_channels(X_train,img_chs)
X_test = copy_to_n_channels(X_test,img_chs)
# transpose according to dimension ordering
if K.image_dim_ordering()=='th':
    print "\nfound the dimension ordering as th"
    X_train = X_train.transpose(0,3,1,2)
    X_test = X_test.transpose(0,3,1,2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

name_gray = False
info = str(name_gray) + "_" + str(img_rows) + "_" + str(img_cols)+ "_" + str(img_chs)
name_X_train = ("X_train_FER"+info)
name_X_test = ("X_test_FER"+info)
name_Y_train = ("Y_train_FER"+info)
name_Y_test = ("Y_test_FER"+info)
# In[18]:

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

np.save(name_X_train, X_train)
np.save(name_X_test, X_test)
np.save(name_Y_train, Y_train)
np.save(name_Y_test, Y_test)
print("\nSaving the processed and loaded data as .npy files")


# In[ ]:



