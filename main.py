import numpy as np
from autokeras.image.image_supervised import ImageClassifier

x_train = np.load('/opt/data/kuzushiji/kmnist-train-imgs.npz')['arr_0']
x_test = np.load('/opt/data/kuzushiji/kmnist-test-imgs.npz')['arr_0']
x_train = x_train.reshape(x_train.shape + (1, ))
x_test = x_test.reshape(x_test.shape + (1, ))

y_train = np.load('/opt/data/kuzushiji/kmnist-train-labels.npz')['arr_0']
y_test = np.load('/opt/data/kuzushiji/kmnist-test-labels.npz')['arr_0']

clf = ImageClassifier(verbose=True)
clf.fit(x_train, y_train, time_limit=12 * 60 * 60)


clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
clf.export_autokeras_model("./autokeras_model.bin") # Auto-Kerasで読み込めるモデルを保存
clf.export_keras_model("./keras_model.bin") # Kerasで読み込めるモデルを保存

acc = clf.evaluate(x_test, y_test)