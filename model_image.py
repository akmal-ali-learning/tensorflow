# Use tensorflow to learn a 1-1 function

import glob
import os
import shutil
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

def positional_encoding(x, y):
    pos_encoding = []
    pos_encoding.append(x)
    pos_encoding.append(y)

    for i in range(-1,13):
      pos_encoding.append(np.sin(x * (2**i)))
      pos_encoding.append(np.sin(y * (2**i)))
      pos_encoding.append(np.cos(x * (2**i)))
      pos_encoding.append(np.cos(y * (2**i)))
    # for i in range(12):
    #   pos_encoding.append(np.mod(x, 1.0/(1.1**i+1)))
    #   pos_encoding.append(np.mod(y, 1.0/(1.1**i+1)))
    return pos_encoding

def get_training_features(image):
    training_features = []
    training_values = []
    width = image.width
    height = image.height
    for y in range(height):
      for x in range(width):
          training_features.append(positional_encoding(y/height,x/width))
    training_features = np.array(training_features)
    training_values = np.reshape(np.asarray(image).copy(), (height*width, 3 ))
    return training_features, training_values



def predict_image(model, height, width, filename ):
  predict_features = []
  for y in range(height):
    for x in range(width):
      predict_features.append(positional_encoding(y/height,x/width))
  predict_features = np.array(predict_features)
  prediction = np.clip(model.predict(predict_features), 0 , 255)
  prediction = np.reshape(prediction, (height, width, 3))
  predicted_image = Image.fromarray((prediction).astype(np.uint8))
  predicted_image.save(filename)
  predicted_image.save("predicted.png")

  
def main():
  results_dir = "results"
  shutil.rmtree(results_dir, ignore_errors=True)
  os.makedirs(results_dir)
  image = Image.open("photo.jpg")
  x_data, y_data = get_training_features(image)

  x_train = x_data.copy()
  y_train = y_data.copy()
  
  model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(x_train.shape[1], )),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(3,)
  ])

  print("Compile model")
  model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

  # Incrementally train model
  for i in range(100):
    print("Train {}".format(i))
    model.fit(x_train, y_train, epochs=1)
    print("Predict {}".format(i))
    predict_image(model, image.height, image.width, "{0}/frame{1:04d}.png".format(results_dir,i))

  # filepaths
  fp_in = "{0}/frame*.png".format(results_dir)
  fp_out = "predicted.gif"

  imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
  img = next(imgs)  # extract first image from iterator
  img.save(fp=fp_out, format='GIF', append_images=imgs,
          save_all=True, duration=200, loop=1)


if __name__ == '__main__':
  sys.exit(main())  # next section explains the use of sys.exit