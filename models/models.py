from typing import List
import layers
from layers import Layer
import json
import numpy as np
from optimizers import Optimizer
import optimizers
from tqdm import tqdm

import sys

self_module = sys.modules[__name__]

class Model:
    """
    [DESC]
        Abstract class for model
    """
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def update_weights(self, weights):
        pass

    def predict(self, x):
        return self.forward(x)

    def fit(self, x, y):
        pass

    def score(self, x, y):
        pass

    def __call__(self, x):
        return self.forward(x)
    
    def compile(self,optimizer : Optimizer=optimizers.SGD(), loss : str="mse"):
        if optimizer == "sgd":
            optimizer = optimizers.SGD()
        self.optimizer = optimizer
        self.optimizer.loss_function = loss
    
    def fit(x,y,batch_size=None,epochs=1):
        pass
    
    
    @staticmethod
    def load(path):
        data = json.load(open(path,"r"))
        result = getattr(self_module,data["model_name"])()
        for k,v in data.items():
            if k == "model_name":
                continue
            setattr(result,k,v)
        for i,layer_data in enumerate(data["layers"]):
            layer = getattr(layers,layer_data["layer_name"]).load(layer_data)
            result.layers[i] = layer
        return result
    
    def save(self,path):
        pass

class Sequential(Model):
    """
    [DESC]
        Class for sequential model
    """
    def __init__(self, layers : List[Layer]=[]):
        """
        [DESC]
            Constructor
        [PARAMS]
            layers : List[Layer]
                List of layers
        """
        if type(layers) != list:
            raise TypeError("layers must be a list of layers")
        self.layers = layers
    
    def add(self, layer : Layer):
        """
        [DESC]
            Add a layer to the model
        [PARAMS]
            layer : Layer
                Layer to add
        """
        if len(self.layers) != 0:
            self.layers.append(layer.set_weight_and_bias(self.layers[-1].output_dim))
        else:
            self.layers.append(layer)

    def forward(self, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Forward propagation
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        result = x.copy()
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def update_weights(self, weights):
        pass

    def fit(self,x,y,batch_size=None,epochs=1):
        assert len(x) == len(y)
        batch_size = len(x) if batch_size is None else batch_size
        if batch_size > len(x):
            raise ValueError("Batch size must be smaller than dataset size")
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        history = {}
        for e in range(epochs):
            for b in tqdm(range(0,len(x),batch_size)):
                batch_x = x[b:b+batch_size]
                batch_y = y[b:b+batch_size]
                batch_y_pred = self.forward(batch_x)
                current_loss = self.optimizer.step(self,y_true=batch_y,y_pred=batch_y_pred)
                self.optimizer.update_weights(self)

                history["loss"] = history.get("loss",[]) + [current_loss]

                # check if the last layer is sigmoid and only one units, round it
                if (type(self.layers[-1]) == layers.Sigmoid and self.layers[-1].output_dim == 1) or \
                    (type(self.layers[-1]) == layers.Dense and self.layers[-1].output_dim == 1 and \
                        self.layers[-1].activation == "sigmoid"):
                    batch_y_pred = np.round(batch_y_pred)
                    batch_y_pred = batch_y_pred.astype(int)
                    batch_y = batch_y.flatten()
                    batch_y_pred = batch_y_pred.flatten()
        return history
            


    def score(self, x, y):
        pass

    def save(self, path):
        model_name = self.__class__.__name__
        result_dict = {
            "model_name": model_name,
            "layers" : []
        }
        for l in self.layers:
            result_dict["layers"].append(l.create_dict())
        with open(path, "w") as f:
            json.dump(result_dict, f)

    # def load(self, path):