from scipy.stats import norm
from sklearn import preprocessing, svm

import utility
import pandas as pd
import numpy as np
import datetime

import pickle

df = utility.read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# ADD INTEGER LABEL TO REPRESENT CLASSES
LABEL = 'ActivityEncoded'
le = preprocessing.LabelEncoder()
df[LABEL] = le.fit_transform(df['activity'].values.ravel())
df['ActivityEncoded'] = df['ActivityEncoded'].astype('int')

# NORMALIZE DATA
df['x-axis'] = utility.feature_normalize(df['x-axis'])
df['y-axis'] = utility.feature_normalize(df['y-axis'])
df['z-axis'] = utility.feature_normalize(df['z-axis'])

df = df.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})
"""
# PREPARE TRAINING DATA
df_train = df[(df['user-id'] <= 28) & (df['ActivityEncoded'] == 5)]

# ADG ALGORITHM
mean, std = norm.fit(df_train['x-axis'].to_numpy())
rnd_x = norm.rvs(size=len(df_train.index))
output_x = [round(x*mean + std, 7) for x in rnd_x]

mean, std = norm.fit(df_train['y-axis'].to_numpy())
rnd_y = norm.rvs(size=len(df_train.index))
output_y = [round(y*mean + std, 7) for y in rnd_y]

mean, std = norm.fit(df_train['z-axis'].to_numpy())
rnd_z = norm.rvs(size=len(df_train.index))
output_z = [round(z*mean + std, 7) for z in rnd_z]

output_array = []
for i in range(len(df_train.index)):
    temp_array = [0, 'Other', 0, output_x[i], output_y[i], output_z[i], 0]
    output_array.append(temp_array)

temp_df = pd.DataFrame(np.array(output_array), columns=["user-id", "activity", "timestamp", "x-axis", "y-axis", "z-axis", "ActivityEncoded"])
df_train = df_train.append(temp_df, ignore_index=True)

print(df_train)


train_X = df_train[['x-axis', 'y-axis', 'z-axis']]
train_Y = df_train['ActivityEncoded'].astype('int')

# DEFINE SVM
clf = svm.SVC()

print('Training starts - ' + str(datetime.datetime.now().time()))

clf.fit(train_X, train_Y)
 
pickle.dump(clf, open('walking_trained_model.sav', 'wb'))

print('Training ended - ' + str(datetime.datetime.now().time()))
"""
# PREPARE TEST DATA
df_test = df[(df['user-id'] > 28) & (df['ActivityEncoded'] == 5)]

print(df_test)

test_X = df_test[['x-axis', 'y-axis', 'z-axis']]
# df_test.loc[:, ('ActivityEncoded')] = 0
test_Y = df_test['ActivityEncoded'].astype('int')

print(test_X)
print(test_Y)

loaded_model = pickle.load(open('walking_trained_model.sav', 'rb'))

xd2 = loaded_model.score(test_X, test_Y)
print(xd2)
