from typing import Tuple
from activation import *
import math

import sys

self_module = sys.modules[__name__]

available_activation_function = {
    "linear": linear,
    "sigmoid": sigmoid,
    "relu": relu,
    "softplus": softplus,
    "tanh": tanh,
}

class Layer:
    """
    [DESC]
        Class for layer
    """
    def __init__(self):
        self.output_dim = None
        self.hidden_state = None

    def forward(self, x : np.ndarray) -> np.ndarray:
        return x

    def update_weights(self, weights):
        pass

    def __call__(self, x : np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def set_weight_and_bias(self, input_dim : int):
        """
        [DESC]
            Set weights and bias
        [PARAMS]
            input_dim : int
                Input units
        """
        if self.output_dim == None:
            self.output_dim = input_dim
        return self
    
    def _error_term(self,error_term_k,w_k,is_last=False,is_mean=True):
        pass
    
    def create_dict(self):
        layer_name = self.__class__.__name__
        variables = vars(self)
        for k,v in variables.items():
            if type(v) == np.ndarray:
                variables[k] = v.tolist()
        variables["layer_name"] = layer_name
        return variables
    
    @staticmethod
    def load(dictionary):
        result = getattr(self_module,dictionary["layer_name"])()
        for k,v in dictionary.items():
            if k == "layer_name":
                continue
            if isinstance(v,list):
                v = np.array(v)
            setattr(result,k,v)
        return result

class Dense(Layer):
    """
    [DESC]
        Class for dense layer
    """
    def __init__(self, units : int=32, activation : str="linear", input_shape : Tuple=None, random_seed : int=None):
        """
        [DESC]
            Constructor
        [PARAMS]
            input_shape : int
                Input size
            units : int
            activation : str
                Activation function
        """
        super().__init__()
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.output_dim = units
        self.random_seed = random_seed
        if input_shape is not None:
            np.random.seed(random_seed)
            self._init_weight_and_bias()

    def _init_weight_and_bias(self):
        """
        [DESC]
            Initialize weights and bias
        """
        self.weights = np.random.randn(self.input_shape[0], self.units)
        self.bias = np.random.randn(self.units)
    
    def set_weight_and_bias(self, input_dim : int):
        """
        [DESC]
            Set weights and bias
        [PARAMS]
            input_dim : int
                Input units
        """
        try:
            np.random.seed(self.random_seed)
            self.weights = np.random.randn(input_dim, self.units)
            self.input_shape = input_dim  #(input_dim,self.units)
        except Exception as e:
            print(input_dim)
            print(self.units)
            raise e
        self.bias = np.random.randn(self.units)
        return super().set_weight_and_bias(input_dim)

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
        len_x_shape = len(x.shape)
        if len_x_shape == 1:
            x = x.reshape((self.input_shape, 1))
        elif len_x_shape != 2 and len_x_shape != 1:
            raise ValueError("Input must be a vector or a matrix")
        self.input_shape = x.shape
        if not hasattr(self, "weights"):
            raise ValueError("You must set input_shape before forward propagation")
        try:
            output = np.matmul(x, self.weights) + self.bias
        except Exception as e:
            print("Input dim:",self.input_shape)
            raise e
        self.input = x
        self.hidden_state = output
        return available_activation_function[self.activation](output)

    def __call__(self, x : np.ndarray) -> np.ndarray:
        return self.forward(x)

available_padding_type = ["same", "valid"]

class Conv2D(Layer):
    """
    [DESC]
        Class for convolutional layer
    """
    def __init__(self, filters : int=32, kernel_size : Tuple[int, int]=(3,3), stride : int=1, padding : str="valid", padding_size : int=0, activation : str="linear", input_shape : Tuple[int]=None, random_seed : int=None):
        """
        [DESC]
            Constructor
        [PARAMS]
            filters : int
                Number of filters
            kernel_size : Tuple[int, int]
                Kernel size
            stride : int
                Stride
            padding : str
                Padding
            padding_size : int
                Padding size
            activation : str
                Activation function
            input_shape : Tuple[int, int, int]
                Input shape
        """
        super().__init__()
        if activation not in available_activation_function:
            raise ValueError("activation must be in %s" % str(available_activation_function.keys()))
        if padding not in available_padding_type:
            raise ValueError("padding must be in %s" % str(available_padding_type))
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_size = padding_size if padding == "same" else 0
        # self.activation = available_activation_function[activation]
        self.activation = activation
        self.random_seed = random_seed

        if input_shape != None:
            np.random.seed(random_seed)
            self.set_weight_and_bias(input_shape)
    
    def set_weight_and_bias(self,input_dim):
        self._processing_input_shape(input_dim)
        kernel_height, kernel_width = self.kernel_size
        self.output_shape = self.filters, (self.input_height-kernel_height+2*self.padding_size)//self.stride+1, (self.input_width-kernel_width+2*self.padding_size)//self.stride+1
        self.output_dim = (self.output_shape[1],self.output_shape[2],self.output_shape[0]) # Removed batch size
        np.random.seed(self.random_seed)
        self._init_weight_and_bias()
        return super().set_weight_and_bias(input_dim)
    
    def _processing_input_shape(self,input_shape : Tuple):
        if len(input_shape) == 3:
            self.input_height, self.input_width, self.input_channels = input_shape
            self.batch_size = None
        elif len(input_shape) == 4:
            self.batch_size, self.input_height, self.input_width, self.input_channels = input_shape
        elif len(input_shape) == 2:
            self.input_height, self.input_width = input_shape
            self.input_channels = 1
            self.batch_size = None
        else:
            raise ValueError("input_shape dimension must be 2 or 3 or 4")
        self.input_shape = (self.batch_size, self.input_height, self.input_width, self.input_channels)
    
    def _init_weight_and_bias(self):
        """
        [DESC]
            Initialize weights and bias
        """
        channels = self.input_channels
        kernel_height, kernel_width = self.kernel_size
        self.weights = np.random.randn(channels, self.filters, kernel_height, kernel_width )
        self.bias = np.zeros(self.output_shape)
    
    def _convolution_per_channel(self, i_channel : int, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Convolution per channel
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        if len(x.shape) != 2:
            raise ValueError("input dimension per channel must be 2 (height, width)")
        # Convolution stage, add padding is done in the forward
        height, width = x.shape
        output_channels = self.filters
        result = np.zeros(self.output_shape)
        for i_kernel in range(output_channels):
            kernel = self.weights[i_channel, i_kernel, :, :]
            for i in range(0, height-self.kernel_size[0]+1, self.stride):
                for j in range(0, width-self.kernel_size[1]+1, self.stride):
                    # element wise multiplication
                    window = x[i:i+self.kernel_size[0], j:j+self.kernel_size[1]]
                    result[i_kernel, i//self.stride, j//self.stride] += np.sum(window * kernel)
        return result
    
    def _convolution_per_entry(self, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Forward propagation per entry
        [PARAMS]
            x : np.ndarray
                Input numpy matrix
        [RETURN]
            np.ndarray
                Output numpy matrix
        """
        if len(x.shape) != 3:
            raise ValueError("input dimension per entry must be 3 (channels, height, width)")
        padded_x = self.add_padding_per_entry(x)
        channels = padded_x.shape[0]
        output = np.zeros(self.output_shape)
        for i in range(channels):
            output += self._convolution_per_channel(i,padded_x[i, :, :])
        # add bias
        output += self.bias
        return output

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
        if len(x.shape) != 4:
            raise ValueError("input dimension must be 4 (batch_size, channels, height, width)")
        if self.batch_size != None:
            if x.shape[0] != self.batch_size:
                raise ValueError("Input's batch_size must be %d" % self.batch_size)
        output = [self._convolution_per_entry(x[i, :, :, :]) for i in range(x.shape[0])]
        output = np.array(output)
        self.input = x
        self.hidden_state = output
        # output = self.activation(output)
        output = available_activation_function[self.activation](output)
        return output

    def add_padding_per_entry(self, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Add padding
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        # Add zeros to the left, up, right, and down according to the padding size
        channels, height, width = x.shape
        result = np.zeros((channels, height+2*self.padding_size, width+2*self.padding_size))
        result[:, self.padding_size:height+self.padding_size, self.padding_size:width+self.padding_size] = x
        return result

# make class activation general
class Activation(Layer):
    """
    [DESC]
        Class for general activation
    """
    def __init__(self, activation : str="relu"):
        super().__init__()
        self.activation = activation
        # self.activation = available_activation_function[activation]
    
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
        # output = self.activation(x)
        output = available_activation_function[self.activation](x)
        return output

# make class activation relu
class ReLU(Layer):
    """
    [DESC]
        Class for relu activation
    """
    def __init__(self):
        super().__init__()
        # self.activation = available_activation_function["relu"]
        self.activation = "relu"

    # fungsi forward relu
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
        # return self.activation(x)
        return available_activation_function[self.activation](x)

# make class activation sigmoid
class Sigmoid(Layer):
    """
    [DESC]
        Class for sigmoid activation
    """
    def __init__(self):
        super().__init__()
        # self.activation = available_activation_function["sigmoid"]
        self.activation = "sigmoid"

    # fungsi forward sigmoid
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
        # return self.activation(x)
        return available_activation_function[self.activation](x)

    # fungsi forward general
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
        # return self.activation(x)
        return available_activation_function[self.activation](x)

class Pooling(Layer):
    # mode='MAX'
    # stride=0
    # filter_size_x=0
    # filter_size_y=0
    def __init__(self,stride:int=0,filter_size:Tuple[int,int]=(3,3),padding:int=0,input_shape: Tuple[int]=None,mode='MAX') :
        super().__init__()
        self.mode = mode
        self.stride = stride
        filter_size_y,filter_size_x=filter_size # ini kan x itu kolom y itu baris
        self.filter_size_x = filter_size_x
        self.filter_size_y = filter_size_y
        self.padding = padding         
        if input_shape != None:
            self.set_weight_and_bias(input_shape)
        else:
            self.input_shape=None
        
    def set_weight_and_bias(self,input_dim):
        self._processing_input_shape(input_dim)
        self.output_shape = self.channel, (self.height-self.filter_size_y+2*self.padding)//self.stride+1, (self.width-self.filter_size_x+2*self.padding)//self.stride+1    
        self.output_dim = (self.output_shape[1],self.output_shape[2],self.output_shape[0])      
        return super().set_weight_and_bias(input_dim)
       
    def _processing_input_shape(self,input_shape : Tuple):
        if len(input_shape) == 3:
            self.height, self.width, self.channel = input_shape
            self.batch_size = None
        elif len(input_shape) == 4:
            self.batch_size, self.height, self.width, self.channel = input_shape
        elif len(input_shape) == 2:
            self.height, self.width = input_shape
            self.channel = 1
            self.batch_size = None
        else:
            raise ValueError("input_shape dimension must be 2 or 3 or 4")
        self.input_shape = (self.height, self.width, self.channel)

    def add_padding_per_channel(self, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Add padding
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        # Add zeros to the left, up, right, and down according to the padding size
        height, width = x.shape
        result = np.zeros((height+2*self.padding, width+2*self.padding))
        result[self.padding:height+self.padding, self.padding:width+self.padding] = x
        return result
    
    def _pool_per_channel(self, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Pooling
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        result = np.zeros((self.output_shape[1], self.output_shape[2]))
        for i in range(0, x.shape[0]-self.filter_size_y+1, self.stride):
            for j in range(0, x.shape[1]-self.filter_size_x+1, self.stride):
                if self.mode == 'MAX':
                    result[i//self.stride, j//self.stride] = np.max(x[i:i+self.filter_size_y, j:j+self.filter_size_x])
                elif self.mode == 'AVERAGE':
                    result[i//self.stride, j//self.stride] = np.mean(x[i:i+self.filter_size_y, j:j+self.filter_size_x])
                else:
                    raise ValueError("mode must be MAX or AVERAGE")
        return result
    
    def _pool_per_entry(self, x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Pooling
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        result = np.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            # padding
            padded_x_i = self.add_padding_per_channel(x[i,:,:])
            pooled_x_i = self._pool_per_channel(padded_x_i)
            result[i, :, :] = pooled_x_i
        return result
        

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
        if len(x.shape) != 4:
            raise ValueError("input dimension must be 4 (batch_size, channels, height, width)")
        if self.batch_size != None:
            if x.shape[0] != self.batch_size:
                raise ValueError("Input's batch_size must be %d" % self.batch_size)
        output = [self._pool_per_entry(x[i, :, :, :]) for i in range(x.shape[0])]
        output = np.array(output)
        self.input = x
        self.hidden_state = output
        return output  
    
    def _inverse_pooling_error_term_per_feature_map(self,error_term,is_mean=False):
        result_dimension = (self.stride * (error_term.shape[0] - 1) - 2 * self.padding \
             + self.filter_size_x, self.stride * (error_term.shape[1] - 1) - 2 * self.padding + self.filter_size_y)
        result = np.zeros(result_dimension)
        # print("Error term matrix dimension : ",error_term.shape)
        for i in range(error_term.shape[0]):
            for j in range(error_term.shape[1]):
                row1 = i*self.stride
                row2 = i*self.stride+self.filter_size_x
                col1 = j*self.stride
                col2 = j*self.stride+self.filter_size_y
                result[row1:row2,col1:col2] += error_term[i,j]
        return result, result_dimension

class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def set_weight_and_bias(self, input_dim: int):
        if len(input_dim) == 4:
            input_dims = input_dim[1:]
        elif len(input_dim) == 3:
            input_dims = input_dim
        self.input_dim = input_dims
        output_dim = 1
        for dim in list(input_dims):
            output_dim *= dim
        self.output_dim = output_dim 
        return super().set_weight_and_bias(input_dim)

    def forward(self, x : np.ndarray) -> np.ndarray:
        self.input = x
        return x.reshape(x.shape[0],-1)

class LSTM(Layer):
    """
    [DESC]
        Class for LSTM Layer
    """
    def __init__(self,units : int=32,input_shape : Tuple[int]=None, return_sequences : bool=False, random_seed : int=None):
        """
        [DESC]
            Constructor
        [PARAMS]
            units : int
                Number of units
            input_shape : Tuple[int]
                Input shape
            return_sequences : bool
                If True, return the last output in the output sequence, or the full sequence if return_state is True.
        """
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        if isinstance(input_shape, tuple):
            self.input_shape = input_shape if len(input_shape) == 2 else (input_shape[1],input_shape[2])
            self.output_dim = units if not return_sequences else (input_shape[0],units)
        else:
            self.input_shape = input_shape
            self.output_dim = units if not return_sequences else (input_shape,units)
        self.input = None
        self.hidden_state = np.zeros((self.units,))
        self.cell_state = np.zeros((self.units,))
        self.output = self.units
        self.random_seed = random_seed
    
    def set_weight_and_bias(self, input_dim: Tuple[int]):
        np.random.seed(self.random_seed)
        if len(input_dim) != 2:
            raise ValueError("input dimension must be 2 (seq_length, feature_dimension)")
        feature_dim = input_dim[1]
        self.weights = {
            "Uf" : np.random.randn(feature_dim, self.units),
            "Ui" : np.random.randn(feature_dim, self.units),
            "Uc" : np.random.randn(feature_dim, self.units),
            "Uo" : np.random.randn(feature_dim, self.units),
            "Wf" : np.random.randn(self.units, self.units),
            "Wi" : np.random.randn(self.units, self.units),
            "Wc" : np.random.randn(self.units, self.units),
            "Wo" : np.random.randn(self.units, self.units),
        }

        self.bias = {
            "bf" : np.random.randn(self.units),
            "bi" : np.random.randn(self.units),
            "bc" : np.random.randn(self.units),
            "bo" : np.random.randn(self.units),
        }
        return super().set_weight_and_bias(input_dim)
    
    def _count_input_gate(self,x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Count input gate
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        it = sigmoid(np.dot(x,self.weights["Ui"]) + np.dot(self.hidden_state,self.weights["Wi"]) + self.bias["bi"])
        return it
    
    def _count_candidate_cell_state(self,x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Count candidate cell state
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        candidate_t = tanh(np.dot(x,self.weights["Uc"]) + np.dot(self.hidden_state,self.weights["Wc"]) + self.bias["bc"])
        return candidate_t
    
    def _count_forget_gate(self,x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Count forget gate
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        ft = sigmoid(np.dot(x,self.weights["Uf"]) + np.dot(self.hidden_state,self.weights["Wf"]) + self.bias["bf"])
        return ft
    
    def _count_output_gate(self,x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Count output gate
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        ot = sigmoid(np.dot(x,self.weights["Uo"]) + np.dot(self.hidden_state,self.weights["Wo"]) + self.bias["bo"])
        return ot
    
    def _count_cell_state(self,ft : np.ndarray,it : np.ndarray,candidate_t : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Count cell state
        [PARAMS]
            ft : np.ndarray
                Forget gate
            it : np.ndarray
                Input gate
            candidate_t : np.ndarray
                Candidate cell state
        [RETURN]
            np.ndarray
                Output
        """
        ct = ft * self.cell_state + it * candidate_t
        self.cell_state = ct
        return ct
    
    def _count_hidden_state(self,ot : np.ndarray,ct : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Count hidden state
        [PARAMS]
            ot : np.ndarray
                Output gate
            ct : np.ndarray
                Cell state
        [RETURN]
            np.ndarray
                Output
        """
        ht = ot * tanh(ct)
        self.hidden_state = ht
        return ht
    
    def forward(self,x : np.ndarray) -> np.ndarray:
        """
        [DESC]
            Forward pass
        [PARAMS]
            x : np.ndarray
                Input
        [RETURN]
            np.ndarray
                Output
        """
        if len(x.shape) == 2:
            x = x.reshape((1,*x.shape))
        self.input = x
        n_batch = x.shape[0]
        output = np.zeros((n_batch,x.shape[1],self.units))
        for t in range(x.shape[1]):
            it = self._count_input_gate(x[:,t])
            candidate_t = self._count_candidate_cell_state(x[:,t])
            ft = self._count_forget_gate(x[:,t])
            ot = self._count_output_gate(x[:,t])
            ct = self._count_cell_state(ft,it,candidate_t)
            ht = self._count_hidden_state(ot,ct)
            output[:,t] = ht
        if self.return_sequences:
            self.output = output
        else:
            self.output = output[:,-1]
        return self.output