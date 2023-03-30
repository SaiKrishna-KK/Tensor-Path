
import matplotlib.pyplot as plt
def load_and_prep(file_loc, img_shape = 224, scale = True):
  import tensorflow as tf
  img = tf.io.read_file(file_loc)
  img = tf.image.decode_jpeg(img)
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    return img/225.
  else:
    return img

def view_random(target_dir, target_class):
  import random, os
  import matplotlib.image as mpimg
  import matplotlib.pyplot as plt
  target_folder = target_dir+target_class
  rand_image = random.sample(os.listdir(target_folder),1)
  img = mpimg.imread(target_folder+'/'+rand_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off')

  print(f'Image Shape: {img.shape}')

def plot_history(history_name):
  import pandas as pd
  model_history = pd.DataFrame(history_name.history)
  model_history.plot(figsize=(11,8), title = 'Model_Performance', xlabel = 'epochs', ylabel = 'performance')


def createModel(model_url, num_classes, img_shape):
  import tensorflow_hub as hub
  import tensorflow as tf
  feature_extractor_layer = hub.KerasLayer(model_url, trainable= False, name= 'feature_extraction_layer', input_shape = img_shape+(3,))
  model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'output_layer')
  ])
  return model

def unzip(file_loc):
  import zipfile
  zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
  zip_ref.extractall()
  zip_ref.close()

def show_path(file_name):
  import os
  for dirpath, dirnames, filenames in os.walk(file_name):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in \'{dirpath}\'')

def print_classes(file_loc):
  import pathlib
  import numpy as np
  data_dir = pathlib.Path(file_loc)
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
  return class_names
