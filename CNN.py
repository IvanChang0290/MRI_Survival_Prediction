import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Concatenate,Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
from skimage.transform import resize
from sklearn.utils.validation import column_or_1d
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import h5py
import re
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

flat_data_arr=[] #input array
target_arr=[] #output array

#file path
path = '/home/archive/BraTS2020_training_data/content/myData'
surInfoFile = "/home/archive/BraTS2020_training_data/content/data/survival_info.csv"

date_column = ['Brats20ID','Age','Survival_days','Extent_of_Resection']
df_surInfo = pd.read_csv(surInfoFile,names= date_column,skiprows=1)


def modify_value(value):
    idStr = str(value)
    number_str = idStr[-3:] 
    number_int = int(number_str)
    return number_int

    
df_surInfo['Brats20ID'] = df_surInfo['Brats20ID'].apply(modify_value)
print(df_surInfo)

pattern = r"volume_(\d+)_"
pattern2 = r"slice_(\d+)"

early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.01, restore_best_weights=True,mode='min')

hidden_layer_list = [30,35,40,45,50,55,60]

for num in hidden_layer_list:
    for i in range(10):
        model = Sequential()
        model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu', input_shape=(155, 240, 240, 4)))
        model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu'))
        model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu'))
        model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(units=num))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
        model.summary()

        y = pd.DataFrame(columns = ['Survival_days','Volume'])
        volumes = []
        #path which contains all the categories of images
        fileCount = 0
        volumeCount = 0
        volumeCountMax = 20
        
        volumeId = -1
        train_num = 200
        count = 0
        
        MSE = 0.0
        dataCount = 0
        dataVolume = []
        volumes = []
        survs = []
        for img in os.listdir(path):
            
            
            if img.endswith(".h5"):
                count += 1
                
                
                fileName = os.path.join(path,img)
                dataId = re.search(pattern, fileName).group(1)
                dataId = int(dataId)
                SliceId = re.search(pattern2, fileName).group(1)
                SliceId = int(SliceId)
        
                if volumeId != dataId :
                    volumeId = dataId
                    volumeCount += 1
                
                if dataId in df_surInfo['Brats20ID'].values:
            
        
                    #print(f'dataId :{dataId}')
                    #print(f'SliceId :{SliceId}')
                    f = h5py.File(fileName, 'r')
                    img_array = f.get('data')
                    img_array = np.array(img_array)
                    dataVolume.append(img_array)
                    
                    fileCount += 1
                    if fileCount % 155 == 0:
                        volumes.append(dataVolume)
                        survDays = df_surInfo.loc[df_surInfo['Brats20ID'] == dataId, 'Survival_days'].iloc[0]
                        survs.append([survDays])
                        dataVolume = []
                    if fileCount == 155*8:
                        print(volumeCount)
                        print(survs)
                        print(np.shape(survs))
                        print(np.shape(volumes))
                        if(volumeCount <= train_num):
                           
                            print('Training')
                            model.fit(np.array(volumes),np.array(survs),epochs=50,callbacks=[early_stopping])
                            print('The Model is trained well with the given images')
                            #print('loss:' + str(loss))
                            #print('m:' + str(mae))
                        
                        else:
                            print('Testing')
                            y_pred=model.predict(np.array(volumes))
                            print("The predicted Data is :")
                            print(y_pred)
                            print("The actual data is:")
                            print(np.array(survs))
                            mse = mean_squared_error(y_pred,survs)
                            MSE += mse
                            dataCount += 1
                            print(f"The model MSE is {mse}")
        
                        volumes = []
                        survs = []
                        fileCount = 0
        with open('output.txt', 'a') as file:
            file.write(str(MSE/dataCount) + '\n')
        file.close()
        print(f"dataCount : {dataCount}")
        print(f"The model MSE is {MSE/dataCount}")

