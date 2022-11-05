import layers
import models
import numpy as np
import image_utils
from sklearn.metrics import f1_score

### This script is debugging script ###

def pooling(input,stride,window_size=(2,2)):
    # output dim = [(W−K+2P)/S]+1, padding 0 in this situation
    output_dim = ((input.shape[0] - window_size[0]) // stride + 1, (input.shape[1] - window_size[1]) // stride + 1)
    output = np.zeros(output_dim)
    for i in range(output_dim[0]):
        for j in range(output_dim[1]):
            output[i,j] = np.max(input[i*stride:i*stride+window_size[0],j*stride:j*stride+window_size[1]])
    return output

def inverse_pooling(output,stride,window_size=(2,2)):
    # input_dim = s * (output_dim - 1) -2p + k
    input_dim = (stride * (output.shape[0] - 1) - 2 * 0 + window_size[0], stride * (output.shape[1] - 1) - 2 * 0 + window_size[1])
    input = np.zeros(input_dim)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            input[i*stride:i*stride+window_size[0],j*stride:j*stride+window_size[1]] += output[i,j]
    return input

def main():
    # Test forward propagation conv2d
    path = "./dataset"
    dataset = image_utils.ImageDataset(path,(32,32))
    x,y = dataset[:]
    model = models.Sequential()
    conv = layers.Conv2D(3,(3,3),1,"same",1,"relu",input_shape=(len(dataset),32,32,3),random_seed=92359234)
    pool = layers.Pooling(2,(2,2),0,mode="AVERAGE")
    model.add(conv)
    print("Conv output dim:", conv.output_dim)
    flatten = layers.Flatten()
    model.add(pool)
    print("Pooling output dim:", pool.output_dim)
    model.add(flatten)
    print("Flatten output dim:", flatten.output_dim)
    dense1 = layers.Dense(12,activation="relu")
    print("Dense1 output dim:", dense1.output_dim)
    model.add(dense1)
    model.add(layers.ReLU())
    print("ReLU output dim:", dense1.output_dim)
    dense2 = layers.Dense(4,activation="relu")
    model.add(dense2)
    print("Dense2 output dim:", dense2.output_dim)
    dense3 = layers.Dense(1,activation="linear")
    model.add(dense3)
    print("Dense3 output dim:", dense3.output_dim)
    model.add(layers.Sigmoid())
    print("Sigmoid output dim:", dense2.output_dim)

    y_pred = model.predict(x)
    print("10 First predictions:", y_pred[:10])
    rounded_y_pred = np.round(y_pred)
    print("F1 score:", f1_score(y,rounded_y_pred))
    print("Value counts of predictions:", np.unique(rounded_y_pred,return_counts=True))

main()