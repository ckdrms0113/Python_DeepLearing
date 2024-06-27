import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

data = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
data.head()

# 마지막 컬럼 전까지는 X로
X = data.values[:,:-1]
# 마지막 컬럼은 Y로
Y = data.values[:,-1]

print('X shape :',X.shape) #=> X shape : (4998, 140)
print('Y shape :',Y.shape) #=> Y shape : (4998,)


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=7, stratify=Y)
print('Train shape :',x_train.shape) #=> Train shape : (3998, 140)
print('Test shape :',x_test.shape) #=> Test shape : (1000, 140)

from collections import Counter
print(Counter(y_train)) #=> Counter({True: 2335, False: 1663})
print(Counter(y_test)) #=> Counter({True: 584, False: 416})

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # test set에는 transform만 사용하기

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

normal_x_train = x_train[y_train]

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9,6))
fig.suptitle("Normal ECG")
ax = ax.ravel() # ax가 3*2차원이기에 for에 1차원으로 만들어 넣기위함
for idx, ax in enumerate(ax):
    ax.grid()
    ax.plot(np.arange(len(normal_x_train[idx])), normal_x_train[idx])
plt.show()

anomalous_x_train = x_train[~y_train]

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9,6))
fig.suptitle("anomalous ECG")
ax = ax.ravel() # ax가 3*2차원이기에 for에 1차원으로 만들어 넣기위함
for idx, ax in enumerate(ax):
    ax.grid()
    ax.plot(np.arange(len(anomalous_x_train[idx])), anomalous_x_train[idx])
plt.show()

x_train_ex = tf.expand_dims(x_train, axis=2)
x_test_ex = tf.expand_dims(x_test, axis=2)

print('원본 차원 정보 :',x_train.shape) #=> 원본 차원 정보 : (3998, 140)
print('변경된 차원 정보 :',x_train_ex.shape) #=> 변경된 차원 정보 : (3998, 140, 1)

# (3998, 140, 1)는 각각 아래의 의미를 가진다.
# (데이터 개수, Sequence 길이, Input Vector 길이)

model = tf.keras.Sequential([
    layers.LSTM(100, return_sequences=True, input_shape=(x_train_ex.shape[1], x_train_ex.shape[2])),
    layers.Dropout(0.25),
    layers.Bidirectional(layers.LSTM(100)),
    layers.Dropout(0.25),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss = 'binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

training_record = model.fit(x_train_ex, y_train,
                    epochs=30,
                    batch_size=128,
                    validation_data=(x_train_ex, y_train),
                    shuffle=True)

fig, ax = plt.subplots()
plt.plot(training_record.history["loss"], label="Training Loss")
plt.plot(training_record.history["val_loss"], label="Validation Loss")
plt.legend()
fig.suptitle("Loss")
plt.show()

pred_proba = model.predict(x_test_ex)

# 2차원인 pred_proba를 1차원으로 변경해준다 (1000,1)=>(1000)
pred_proba_1d = pred_proba.reshape(-1)

# 임계치 이상이면 True 미만이면 False를 부여한다.
threshold = 0.5
pred = (pred_proba_1d >= threshold)

# Compute the metrics
accuracy_test_rnn= accuracy_score(y_test, pred)
print(f'Accuracy: {accuracy_test_rnn}')
#=> Accuracy: 0.988

precision_test_rnn=precision_score(y_test, pred)
print(f'Precision = {round(precision_test_rnn,3)}')
#=> Precision = 0.99

recall_test_rnn=recall_score(y_test, pred)
print(f'Recall = {round(recall_test_rnn,3)}')
#=> Recall = 0.99
