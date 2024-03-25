# This is the code from https://github.com/shalabh147/Brain-Tumor-Segmentation-and-Survival-Prediction-using-Deep-Neural-Networks/
# This model gets a slice of every image type (seg, flair, t1, t1ce, t2) and trains it on the survival data
import sys

import random
import math
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
from keras.layers import Concatenate, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize, rescale
from sklearn.utils import class_weight
from keras.models import Sequential

import nibabel as nib

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import csv

from tqdm import tqdm

# import pickle

original_surv_file_path = '/root/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'

survival_file_path = '/root/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/training_set.csv'
training_data_path = '/root/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
model_save_path = '/root/code/xai-for-brain-img-surv/models/regression_3d_scaled_v1/results_24000/surv_pred_3d_scaled_v1'
age_dict = {}
days_dict = {}
SAMPLES_PER_ITERATION = 400 #300  # set this to define the number of samples for each iteration. Should not be too large, otherwise you'll run out of RAM
BATCH_SIZE = 64 #50  # set the batch size for training in each epoch. If this is too large, your GPU will complain
EPOCHS = 400 #400 #200
RANDOM_ITERATIONS = 25


def printHelp():
  print("Usage: ./train_surv_pred.sh regression_3d [-h] <MRI_TYPE> [-c]")
  print("[-h]: Print short help")
  print("<MRI_TYPE>: Valid types are: flair, t1, t1ce, t2")
  print("[-c] (optional): Continue training model (if it exists)")


# from joblib import dump
if len(sys.argv) < 2:
  print("Error: Argument MRI_TYPE is missing!")
  printHelp()
  exit(1)

if sys.argv[1] == "-h":
  printHelp()
  exit(0)

# check if -c flag is set
continue_training = False
if len(sys.argv) == 3 and sys.argv[2] == "-c":
  continue_training = True

mri_type = sys.argv[1]

if mri_type != "flair" and mri_type != "t1" and mri_type != "t1ce" and mri_type != "t2":
  print("Invalid MRi type: " + mri_type + "; Valid types are: flair, t1, t1ce, t2")
  exit(1)

model_save_path = model_save_path + "_" + mri_type


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


# CNN Structure defined in here
def SurvPredNet(input_img, age_m):
  # input_img = BatchNormalization()(input_img)
  a1 = Conv2D(16, kernel_size=(3, 3), padding='same')(input_img)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  a1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a1)

  a1 = Conv2D(32, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  a1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a1)
  #
  a1 = Conv2D(32, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  a1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a1)

  a1 = Conv2D(48, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)
  #
  a1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a1)

  a1 = Conv2D(64, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  a1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1))(a1)

  a1 = Conv2D(64, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  a1 = MaxPooling2D(pool_size=(6, 6), strides=(1, 1))(a1)
  # a1 = AveragePooling2D(pool_size=(6, 6), strides=(1, 1))(a1)

  a1 = Conv2D(64, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  # a1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(a1)
  # a1 = AveragePooling2D(pool_size=(5, 5), strides=(1, 1))(a1)

  a1 = Conv2D(64, kernel_size=(3, 3), padding='same')(a1)
  a1 = BatchNormalization()(a1)
  a1 = Activation('relu')(a1)

  a1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(a1)

  a1 = Flatten()(a1)
  a1 = Concatenate()([a1, age_m])
  a1 = BatchNormalization()(a1)

  # a1 = Dense(64, activation='relu')(a1)
  a1 = Dense(32, activation='relu')(a1)
  a1 = Dense(16, activation='relu')(a1)
  outputs = Dense(1, activation='sigmoid')(a1)

  model = Model(inputs=[input_img, age_m], outputs=outputs)

  return model


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
    std_dev = np.std(centered)
    if (std_dev != 0):
      centered = centered / std_dev

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



# check if training should be continued
complete_model_filename = model_save_path + ".h5"
if continue_training == True and os.path.exists(complete_model_filename):
  print("Loading existing model from " + complete_model_filename)
  model = load_model(complete_model_filename)
  model.summary()
else:
  # Set Input size (5 layers: one slice of every MRI type + seq) and load model
  input_img = Input((120, 120, 78))  # half the size of the original
  age_m = Input((1,))
  model = SurvPredNet(input_img, age_m)
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
  model.summary()

# load all directories in data path
# all_images = [a for a in os.listdir(training_data_path) if os.path.isdir(os.path.join(training_data_path, a))]
# print(len(all_images))
# all_images.sort()

loss_hist = []
accu_hist = []
epoch_wise_loss = []
epoch_wise_accu = []
# for epochs in range(45):
epoch_loss = 0
epoch_accu = 0


def plotLoss(loss_hist, showPlot=False):
  plt.plot(loss_hist)
  plt.title('Model_loss vs epochs')
  plt.ylabel('Loss')
  plt.xlabel('epochs')
  plt.savefig(model_save_path)
  if showPlot == True:
    plt.show()
  plt.close()


def trainIteration(offset, iteration_size, epochs, batch_size, initial_epoch, randomize=False):
  input_to_model = np.zeros((iteration_size, 120, 120, 78), dtype=np.float16)  # survival_data_count
  age = np.zeros((iteration_size, 1))
  ground_truth = np.zeros((iteration_size, 1))
  cnt = 0

  print("=== Loading Data Points ===")
  for i in tqdm(range(iteration_size)):

    if randomize == False:
      image_num = i + offset
    else:
      image_num = random.randint(0, len(all_images) - 1)

    if image_num >= len(all_images):  # Overflow. Starting from beginning
      image_num = image_num % len(all_images)

    data = np.zeros((120, 120, 78))
    x = all_images[image_num]

    folder_path = training_data_path + '/' + x
    modalities = os.listdir(folder_path)
    modalities.sort()
    # data = []

    for j in range(len(modalities)):  # images get loaded here!
      # print(modalities[j])
      image_path = folder_path + '/' + modalities[j]
      if (image_path.find(mri_type + '.nii') != -1):
        img = nib.load(image_path);
        # print(img)
        image_data = img.get_fdata()
        # print(type(image_data))
        image_data = np.asarray(image_data)
        # print(image_data)
        image_data = rescale(image_data, 0.5, anti_aliasing=False)
        # image_data = resize(image_data, (120, 120, 78))

        image_data = standardize(image_data)
        # print(image_data)
        data = image_data
        # print("Entered modality")
        break

    # print("Loading data point (" + str(cnt + 1) + "/" + str(iteration_size) + "): " + x)
    # print(image_data2.shape)

    input_to_model[cnt] = data
    # age = np.zeros((1,1))
    age[cnt, 0] = float(age_dict[x])
    days = int(days_dict[x])

    ground_truth[cnt, 0] = int(days) / max_survival_days
    # print(ground_truth[cnt])
    cnt += 1

  # y_to = keras.utils.to_categorical(y_to,num_classes=4)

  history = model.fit(x=[input_to_model, age], y=ground_truth, epochs=epochs + initial_epoch, batch_size=batch_size,
                      initial_epoch=initial_epoch, verbose=2)
  # history = model.fit(x=[input_to_model,age],y=ground_truth, epochs = epochs, batch_size = batch_size)
  loss_hist.extend(history.history['loss'])
  model.save(model_save_path + '.h5')
  plotLoss(loss_hist, False)


if survival_data_count < SAMPLES_PER_ITERATION:  # avoid unnecessary overflow if data is too small
  SAMPLES_PER_ITERATION = survival_data_count

# THIS IS WHERE THE MAGIC HAPPENS. TRAINING STARTS HERE
offset_counter = 0
print("== Start Training ==")
# for iteration in range(round(survival_data_count / SAMPLES_PER_ITERATION)):
#   # print("Iteration: %6d | Survival Count: %6d | Samples per iteration: %6d |" % (iteration, survival_data_count, SAMPLES_PER_ITERATION))
#   # print("%5d" % (round(survival_data_count / SAMPLES_PER_ITERATION)))
#   # exit(200)
#   trainIteration(offset_counter, SAMPLES_PER_ITERATION, epochs=EPOCHS, batch_size=BATCH_SIZE,
#                  initial_epoch=int(offset_counter / SAMPLES_PER_ITERATION) * EPOCHS)
#   offset_counter += SAMPLES_PER_ITERATION

while (offset_counter < survival_data_count):
  print("Conducting iteration (" + str(int(offset_counter / SAMPLES_PER_ITERATION) + 1) + "/" + str(
    math.ceil(survival_data_count / SAMPLES_PER_ITERATION)) + ")")
  trainIteration(offset_counter, SAMPLES_PER_ITERATION, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 initial_epoch=int(offset_counter / SAMPLES_PER_ITERATION) * EPOCHS)
  offset_counter += SAMPLES_PER_ITERATION

for i in tqdm(range(RANDOM_ITERATIONS)):  # do some training with random sampling
  # print("Conducting random iteration (" + str(i + 1) + "/" + str(RANDOM_ITERATIONS) + ")")
  trainIteration(0, SAMPLES_PER_ITERATION, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 initial_epoch=int(offset_counter / SAMPLES_PER_ITERATION + i) * EPOCHS, randomize=True)

print(loss_hist)
plotLoss(loss_hist, False)
