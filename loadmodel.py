import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import np_utils

import seaborn as sns
import numpy as np
import os
import utility

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report

print("Trained model loading...")
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_m = model_from_json(loaded_model_json)
# load weights into new model
model_m.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])


df = utility.read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()

# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['activity'].values.ravel())

df_test = df[df['user-id'] > 28]
df_train = df[df['user-id'] <= 28]

df_test['x-axis'] = utility.feature_normalize(df_test['x-axis'])
df_test['y-axis'] = utility.feature_normalize(df_test['y-axis'])
df_test['z-axis'] = utility.feature_normalize(df_test['z-axis'])

df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})


df_train['x-axis'] = utility.feature_normalize(df['x-axis'])
df_train['y-axis'] = utility.feature_normalize(df['y-axis'])
df_train['z-axis'] = utility.feature_normalize(df['z-axis'])

df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test = utility.create_segments_and_labels(df_test,
                                                    80,
                                                    40,
                                                    LABEL)

x_train, y_train = utility.create_segments_and_labels(df_train,
                                                      80,
                                                      40,
                                                      LABEL)
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
input_shape = (num_time_periods*num_sensors)
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

num_classes = le.classes_.size
y_test = np_utils.to_categorical(y_test, num_classes)


y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

utility.show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))
