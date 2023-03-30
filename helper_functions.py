

def load_and_prep(file_loc, img_shape = 224, scale = True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  Parameters
  ----------
  file_loc (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  import tensorflow as tf
  img = tf.io.read_file(file_loc)
  img = tf.image.decode_jpeg(img)
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    return img/225.
  else:
    return img

def view_random(target_dir+'/', target_class):
  """
  Reads in an image from trage directory and the target class and plots a random image. 
  Parameters
  ----------
  target_dir (str): string filename of target directory
  target_class (str): string filename of target directory
 
  """
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
  """
  Takes the name of the Tensorflow history file and us pandas to plot.
  Parameters 
  ----------
  history_name (variable): Variable history that was captured during the model training
  """

  import pandas as pd
  model_history = pd.DataFrame(history_name.history)
  model_history.plot(figsize=(11,8), title = 'Model_Performance', xlabel = 'epochs', ylabel = 'performance')


def createModel(model_url, num_classes, img_shape):
  """
  Take the model from the model_url(from the tensorhub) and create a model using those weights and biases. 
  However the weights and biases are frozen, only the final layer weights and biases are trained. 
  Finally return the created model with information from the input parameters.
  Parameters
  ----------
  model_url (str): string model url from the tensorhub
  num_classes (int): Number of classes in the output layer
  img_shape (tupple): Input Image shape given to the transfer learning model
  """

  import tensorflow_hub as hub
  import tensorflow as tf
  feature_extractor_layer = hub.KerasLayer(model_url, trainable= False, name= 'feature_extraction_layer', input_shape = img_shape+(3,))
  model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'output_layer')
  ])
  return model

def unzip(file_loc):
  """
  unzips the given file location.
  Parameters
  ----------
  file_loc (str): String file location path
  """
  import zipfile
  zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
  zip_ref.extractall()
  zip_ref.close()

def show_path(file_name):
  """
  Prints all the directories, file names and number files exist in the leaf file.
  Parameters
  ----------
  file_name (str): String file name
  """
  import os
  for dirpath, dirnames, filenames in os.walk(file_name):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in \'{dirpath}\'')

def print_classes(file_loc):
  """
  Prints all the class names, when the data is stored in the form of directories
  ----------
  file_loc(str): String file location path
  """
  import pathlib
  import numpy as np
  data_dir = pathlib.Path(file_loc)
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
  return class_names
