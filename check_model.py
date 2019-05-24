from tensorflow import keras
from keras.utils import plot_model

model = keras.models.load_model("./keras_model.bin")
plot_model(model, to_file='model.png')