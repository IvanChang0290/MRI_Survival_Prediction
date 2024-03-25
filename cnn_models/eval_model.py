# This is the code from https://github.com/shalabh147/Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks/
# This model gets a slice of every image type (seg, flair, t1, t1ce, t2) and trains it on the survival data

import sys

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorrt as trt
import tensorflow as tf
import keras.backend as K
# import keras
# from ensorflow import keras
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Maximum, Flatten
from keras.layers import Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, UpSampling2D
# from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
# from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, MaxPooling3D, AveragePooling2D
from keras.layers import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from sklearn.utils import class_weight
from keras.models import Sequential
import nibabel as nib
from skimage.transform import resize, rescale
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import csv

# import pickle

# from joblib import dump


original_surv_file_path = '/root/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
survival_file_path = '/root/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/test_set.csv'#gtr_test_set.csv
training_data_path = '/root/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'


prediction_model_path = '/root/code/xai-for-brain-img-surv/models/regression_3d_scaled_v1/results_24000/surv_pred_3d_scaled_v1'
age_dict = {}
days_dict = {}

if len(sys.argv) < 2:
  print("Error: Argument MRI_TYPE is missing!")
  print("Usage: ./eval_surv_pred.sh regression_3d <MRI_TYPE>. Valid types are: flair, t1, t1ce, t2")
  exit(1)

mri_type = sys.argv[1]

if mri_type != "flair" and mri_type != "t1" and mri_type != "t1ce" and mri_type != "t2":
  print("Invalid MRi type: " + mri_type + "; Valid types are: flair, t1, t1ce, t2")
  exit(1)

prediction_model_path = prediction_model_path + "_" + mri_type


def dice_coef(y_true, y_pred, epsilon=0.00001):
  """
  Dice = (2*|X & Y|)/ (|X|+ |Y|)
       =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
  ref: https://arxiv.org/pdf/1606.04797v1.pdf

  """
  axis = (0, 1, 2)
  dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
  dice_denominator = K.sum(y_true * y_true, axis=axis) + K.sum(y_pred * y_pred, axis=axis) + epsilon
  return K.mean((dice_numerator) / (dice_denominator))


def dice_coef_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred)


# returns centered slice of 3D input image
def standardize(image):
  standardized_image = np.zeros(image.shape)
  # iterate over the `z` dimension
  for z in range(image.shape[2]):
    # get a slice of the image
    # at channel c and z-th dimension `z`
    image_slice = image[:, :, z]

    # subtract the mean from image_slice
    centered = image_slice - np.mean(image_slice)

    # divide by the standard deviation (only if it is different from zero)
    if (np.std(centered) != 0):
      centered = centered / np.std(centered)

    # update  the slice of standardized image
    # with the scaled centered and scaled image
    standardized_image[:, :, z] = centered

  return standardized_image


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Read survival infos
survival_data_count = 0
max_survival_days = 0
all_images = []
with open(survival_file_path, mode='r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  a = 0
  b = 0
  c = 0
  first = True
  for row in csv_reader:
    if first == True:
      print(f'Column names are {", ".join(row)}')
      first = False
    else:
      print(row)
      key = row[0]
      age = row[1]
      days = row[2]
      age_dict[key] = float(age)
      days_dict[key] = int(days)
      #max_survival_days = max(max_survival_days, int(days))
      all_images.append(key)
      if int(days) < 250:
        a += 1
      elif (int(days) >= 250 and int(days) <= 500):
        b += 1
      else:
        c += 1
      survival_data_count += 1

  print(f'Processed {survival_data_count} survival data points.')
  # age_m = np.zeros((1,1))
  print(a, b, c)
  #print(max_survival_days)

with open(original_surv_file_path, mode='r') as csv_file2:
  csv_reader = csv.reader(csv_file2, delimiter=',')
  first = True
  for row in csv_reader:
    if first == True:
      #print(f'Column names are {", ".join(row)}')
      first = False
    else:
      key = row[0]
      age = row[1]
      days = row[2]
      max_survival_days = max(max_survival_days, int(days))

  print(max_survival_days)



# Load prediction model
surv_model = load_model(prediction_model_path + ".h5")
# all_images.sort()

epoch_loss = 0
epoch_accu = 0
input_to_model = np.zeros((1, 120, 120, 78), dtype=np.float16)
age = np.zeros((1, 1))
ground_truth = 0.0
cnt = 0
classification_count = 0.0
sq_err = 0.0
fig_x = []
fig_acc = []

f = open(prediction_model_path + "_training.csv", mode="a")
for image_num in range(len(all_images)):
  data = np.zeros((120, 120, 78))
  x = all_images[image_num]

  folder_path = training_data_path + '/' + x;
  modalities = os.listdir(folder_path)
  modalities.sort()

  for j in range(len(modalities)):  # images get loaded here!

    image_path = folder_path + '/' + modalities[j]
    if (image_path.find(mri_type + '.nii') != -1):
      img = nib.load(image_path);
      image_data = img.get_fdata()
      image_data = np.asarray(image_data)
      image_data = rescale(image_data, 0.5, anti_aliasing=False)
      # image_data = resize(image_data, (120, 120, 78))
      image_data = standardize(image_data)
      data = image_data

      break

  # print("Loading data point (" + str(cnt + 1) + "/" + str(len(all_images)) + "): " + x)
  # print(image_data2.shape)

  input_to_model[0] = data
  age[0, 0] = float(age_dict[x])
  days = int(days_dict[x])
  ground_truth = int(days) / max_survival_days
  cnt += 1

  # score = surv_model.evaluate(x = [input_to_model,age], y = ground_truth)
  pred = surv_model.predict(x=[input_to_model, age])
  gt_value = ground_truth * max_survival_days
  pred_value = pred * max_survival_days
  if gt_value < 300 and pred_value < 300:
    classification_count += 1
  elif gt_value <= 450 and pred_value <= 450:
    classification_count += 1
  elif gt_value > 450 and pred_value > 450:
    classification_count += 1

  diff = abs(gt_value - pred_value)
  sq_err += (diff * diff)
  divisor = image_num + 1

  print(f'Ground trught: {gt_value} | Prediction: {pred_value} | Diff : {diff} | Acc: {(classification_count / divisor)} | MSE: {sq_err / divisor} |')
  f.write(x + "," + str(int(pred_value)) + "\n")
f.close()
  # print(
  #   "Ground truth: " + str(gt_value) + "; Prediction: " + str(pred_value) + "; Diff: " + str(diff) + "; Acc: " + str(
  #     classification_count / divisor) + "; MSE: " + str(sq_err / divisor))


print(
  "Final Accuracy: " + str(classification_count / survival_data_count) + "; MSE: " + str(sq_err / survival_data_count))
