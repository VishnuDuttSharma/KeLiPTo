import h5py
import json
import numpy as np

def get_config(filename):
    file_pt = h5py.File(filename)
    arch_dict = json.loads(file_pt.attrs['model_config'].decode('utf-8'), encoding='utf-8')
    return arch_dict['config']

def linear(x):
    return x
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def hard_sigmoid(self, x):
    return max(0, min(1, x*0.2 + 0.5))
def softmax(x):
    z_exp = [np.exp(i) for i in x]
    return np.array([i / sum_z_exp for i in z_exp])
def relu(x):
    return max(0, x)

activation_dict = {
    'linear' : linear,
    'tanh' : tanh,
    'sigmoid' : sigmoid,
    'hard_sigmoid' : hard_sigmoid,
    'relu': relu
}
    
class Layer:
    def __init__(self, config={}):
        self.class_name = config['class_name']
        self.name = config['name']
        self.inbound_nodes = config['inbound_nodes']
        self.trainable_weights=None
        
    def load_from_dataset(self, h5_dataset):
        for ind,name in enumerate(h5_dataset[self.name].attrs['weight_names']):
            self.trainable_weights[ind] = h5_dataset[self.name][name].values

class Input(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        
class Embedding(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        in_config = config['config']
        self.input_dim = in_config['input_dim']
        self.output_dim = in_config['output_dim']
        self.dtype = in_config['dtype']
        self.kernel = None
        self.trainable_weights = [self.kernel]
    
    def load_weights(self, wgt=[]):
        self.kernel = wgt.value
        
    def __call__(self, input_vec):
        return self.kernel[input_vec]

class RNN(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        in_config = config['config']
        self.activation = in_config['activation']
        self.recurrent_activation = in_config['recurrent_activation']
        self.return_sequences = in_config['return_sequences']
        self.units = in_config['units']
        self.use_bias = in_config['use_bias']
        self.kernel = None
        self.recurrent_kernel = None
        self.bias= None
        if self.use_bias:
            self.trainable_weight = [self.kernel, self.recurrent_kernel, self.bias]
        else:
            self.trainable_weight = [self.kernel, self.recurrent_kernel]
    
    def load_weights(self, wgt_list=[]):
        self.kernel = wgt_list[0]
        self.recurrent_kernel = wgt_list[1]
        if self.use_bias:
            self.bias = wgt_list[1]
        else:
            self.bias = np.zeros(shape=(self.units,))
        
    def __call__(self, input_vec):
        timesteps = input_vec[-2]
        h_t  = np.zeros(shape(timesteps, self.units))
        x_w_b = np.dot(input_vec, self.kernel) + self.bias
        for tm in range(timesteps):
            h_t[i] = self.activation( x_w_b[tm] + np.dot(h_t[tm-1], self.recurrent_kernel))
        if self.return_sequences:
            return h_t
        else:
            return h_t[-1]

class LSTM(RNN):
    def __init__(self, config={}):
        super().__init__(config)
    
    def load_weights(self, wgt_list=[]):
        super().load_weights(wgt_list)

    def __call__(self, input_vec):
        timesteps = input_vec[-2]
        h_t = np.zeros(shape=(timesteps, self.units))
        C_t = np.zeros(shape=(self.units,))
        x_w_b = np.dot(input_vec, self.kernel) + self.bias
        for tm in range(timesteps):
            mat_out = x_w_b[tm] + np.dot(h_t[tm-1], self.recurrent_kernel)
            
            i = self.recurrent_activation(mat_out[:, :self.units])
            f = self.recurrent_activation(mat_out[:, self.units: self.units * 2])
            c = f * c_tm1 + i * self.activation(mat_out[:, self.units * 2: self.units * 3])
            o = self.recurrent_activation(mat_out[:, self.units * 3:])
            
            h_t[tm] = o * self.activation(c)
        
        if self.return_sequences:
            return h_t
        else:
            return h_t[-1]

class GRU(RNN):
    def __init__(self, config={}):
        super().__init__(config)
    
    def load_weights(self, wgt_list=[]):
        super().load_weights(wgt_list)

    def __call__(self, input_vec):
        timesteps = input_vec[-2]
        h_t = np.zeros(shape=(timesteps, self.units))
        h_bar = np.zeros(shape=(self.units,))
        x_w_b = np.dot(input_vec, self.kernel) + self.bias
        
        for tm in range(timesteps):
            mat_out = x_w_b[tm] + np.dot(h_t[tm-1], self.recurrent_kernel)
            z = self.activation(mat_out[:self.units])
            r = self.activation(mat_out[self.units: self.units * 2])
            h_bar = self.recurrent_activation(mat_out[self.units * 2:])
            h_t[tm] = (1 - z) * h_t[-1] + z * h_bar
        
        if self.return_sequences:
            return h_t
        else:
            return h_t[-1]

class Dense(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        in_config = config['config']
        self.activation = in_config['activation']
        self.units = in_config['units']
        self.use_bias = in_config['use_bias']
        self.kernel = None
        self.bias= None
        if self.use_bias:
            self.trainable_weight = [self.kernel, self.bias]
        else:
            self.trainable_weight = [self.kernel]
    
    def load_weights(self, wgt_list=[]):
        self.kernel = wgt_list[0]
        if self.use_bias:
            self.bias = wgt_list[1]
        else:
            self.bias = np.zeros(shape=(self.units,))
    
    def __call__(self, input_vec):
        return self.activation(np.dot(input_vec, self.kernel) + self.bias)

