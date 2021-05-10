import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
from matplotlib import pyplot as plt
import utility
from sklearn import preprocessing

print("before load")
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



def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

df = utility.read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['activity'].values.ravel())


# Differentiate between test set and training set
df_test = df[df['user-id'] > 28]
x_test, y_test = utility.create_segments_and_labels(df_test,
                                             80,
                                             40,
                                             LABEL)

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))

