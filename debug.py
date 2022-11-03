from models import *
from layers import *

import numpy as np

def debug1():
    model = Sequential()
    dense1 = Dense(units=32, input_shape=(20,), activation='relu')
    model.add(dense1)
    dense2 = Dense(units=32, activation='relu')
    model.add(dense2)
    sigmoid = Dense(units=1, activation='sigmoid')
    model.add(sigmoid)

    data = np.random.random((10, 20))

    print(model.predict(data))

def debug2():
    lstm = LSTM(units=5, input_shape=(5, 20, 10), return_sequences=False, random_seed=0)

    lstm.set_weight_and_bias((20,10))

    x = np.random.random((5,20,10))

    # print(lstm.weights)
    # print("X : ", x)

    print(lstm.forward(x))

if __name__ == "__main__":
    # debug1()
    debug2()