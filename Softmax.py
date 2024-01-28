import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
#matplotlib widget
from matplotlib.widgets import Slider
#from lab_utils_common import dlc
#from lab_utils_softmax import plt_softmax
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# tf.autograph.set_verbosity(0)


def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)

# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
#两千个数据，没个数据可以划分为4个类别，每个数据有两个特征

# x_train的形状
print(X_train.shape) #训练数据集的输入特征矩阵。每行代表一个样本，每列代表一个特征。
print(y_train.shape) #y 代表4个类别 （0，1，2，3）

model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
        
p_nonpreferred = model.predict(X_train) # 形状是（2000，4）
print(p_nonpreferred.shape)
print(p_nonpreferred [:2]) #取出前两个样本
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))

preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=15
)
        
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))
print("\n")

sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))
print("\n")


for i in range(10):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")