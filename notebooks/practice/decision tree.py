import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2
from sklearn.feature_extraction import DictVectorizer
import sklearn.model_selection as ms
from sklearn import metrics
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

plt.style.use('ggplot')
plt.switch_backend('agg')

data = [
    {'age': 33, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high',
     'Na': 0.66, 'K': 0.06, 'drug': 'A'},
    {'age': 77, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal',
     'Na': 0.19, 'K': 0.03, 'drug': 'D'},
    {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal',
     'Na': 0.80, 'K': 0.05, 'drug': 'B'},
    {'age': 39, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal',
     'Na': 0.19, 'K': 0.02, 'drug': 'C'},
    {'age': 43, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'high',
     'Na': 0.36, 'K': 0.03, 'drug': 'D'},
    {'age': 82, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal',
     'Na': 0.09, 'K': 0.09, 'drug': 'C'},
    {'age': 40, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal',
     'Na': 0.89, 'K': 0.02, 'drug': 'A'},
    {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal',
     'Na': 0.80, 'K': 0.05, 'drug': 'B'},
    {'age': 29, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal',
     'Na': 0.35, 'K': 0.04, 'drug': 'D'},
    {'age': 53, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal',
     'Na': 0.54, 'K': 0.06, 'drug': 'C'},
    {'age': 63, 'sex': 'M', 'BP': 'low', 'cholesterol': 'high',
     'Na': 0.86, 'K': 0.09, 'drug': 'B'},
    {'age': 60, 'sex': 'M', 'BP': 'low', 'cholesterol': 'normal',
     'Na': 0.66, 'K': 0.04, 'drug': 'C'},
    {'age': 55, 'sex': 'M', 'BP': 'high', 'cholesterol': 'high',
     'Na': 0.82, 'K': 0.04, 'drug': 'B'},
    {'age': 35, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'high',
     'Na': 0.27, 'K': 0.03, 'drug': 'D'},
    {'age': 23, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high',
     'Na': 0.55, 'K': 0.08, 'drug': 'A'},
    {'age': 49, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal',
     'Na': 0.27, 'K': 0.05, 'drug': 'C'},
    {'age': 27, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal',
     'Na': 0.77, 'K': 0.02, 'drug': 'B'},
    {'age': 51, 'sex': 'F', 'BP': 'low', 'cholesterol': 'high',
     'Na': 0.20, 'K': 0.02, 'drug': 'D'},
    {'age': 38, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal',
     'Na': 0.78, 'K': 0.05, 'drug': 'A'}
]
print(len(data))
# select 'drug' attribute as target and remove it from data
target = [dic.pop('drug') for dic in data]

sodium = [d['Na'] for d in data]
potassium = [d['K'] for d in data]

plt.scatter(sodium, potassium)
plt.xlabel('sodium')
plt.ylabel('potassium')

# dots with different colors
target_flags = [ord(i) - 65 for i in target]
print('target_flags =', target_flags)
plt.scatter(sodium, potassium, c=target_flags, s=100)
plt.xlabel('sodium')
plt.ylabel('potassium')

age = [d['age'] for d in data]
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.scatter(sodium, potassium, c=target_flags, s=100)
plt.xlabel('sodium (Na)')
plt.ylabel('potassium (K)')
plt.subplot(2, 2, 2)
plt.scatter(age, sodium, c=target_flags, s=100)
plt.xlabel('age')
plt.ylabel('sodium (Na)')
plt.subplot(2, 2, 3)
plt.scatter(age, potassium, c=target_flags, s=100)
plt.xlabel('age')
plt.ylabel('potassium (K)')

### Preprocessing the data
# convert categorical data to numerical
vec = DictVectorizer(sparse=False)
data_pre = vec.fit_transform(data)
print('get_feature_names() =', vec.get_feature_names())
# convert to float32 to compatible with OpenCV
data_pre = np.array(data_pre, dtype=np.float32)
target_flags = np.array(target_flags, dtype=np.float32)
# split train & test datasets
x_train, x_test, y_train, y_test = ms.train_test_split(data_pre, target_flags, test_size=5, random_state=42)

### Constructing the decision treex
dtree = cv2.ml_DTrees.create()
print('dtree created')
# train the tree
dtree.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
print('dtree.train complete')

y_pred = dtree.predict(x_test)
print('y_pred =', y_pred)
metrics.accuracy_score(y_test, y_pred)
