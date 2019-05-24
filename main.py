#coding: utf8
import numpy as np
import cv2
import os
from autokeras.image.image_supervised import ImageClassifier

x_train = []
y_train = []
x_test = []
y_test = []


for file_name in  os.listdir("train/normal"):
    img = cv2.imread("train/normal/"+file_name)
    x_train.append(img)
    # x_train.reshape(256,256,3)
    y_train.append(0)

for file_name in  os.listdir("train/anomaly"):
    img = cv2.imread("train/anomaly/"+file_name)
    x_train.append(img)
    # x_train.reshape(256,256,3)
    y_train.append(0)
x_train = np.array(x_train)
y_train = np.array(y_train)

for file_name in  os.listdir("test/normal"):
    img = cv2.imread("test/normal/"+file_name)
    x_test.append(img)
    # x_train.reshape(256,256,3)
    y_test.append(0)

for file_name in  os.listdir("test/anomaly"):
    img = cv2.imread("test/anomaly/"+file_name)
    x_test.append(img)
    # x_train.reshape(256,256,3)
    y_test.append(0)
x_test = np.array(x_test)
y_test = np.array(y_test)


print(x_train.shape)
print(y_train.shape)

clf = ImageClassifier(verbose=True)
clf.fit(x_train, y_train, time_limit=12 * 60 * 60)


clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
clf.export_autokeras_model("./autokeras_model.bin") # Auto-Kerasで読み込めるモデルを保存
clf.export_keras_model("./keras_model.bin") # Kerasで読み込めるモデルを保存

acc = clf.evaluate(x_test, y_test)