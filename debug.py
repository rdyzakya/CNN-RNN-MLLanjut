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

if __name__ == "__main__":
    debug1()