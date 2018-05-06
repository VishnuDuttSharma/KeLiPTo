import h5py
import json
import numpy as np

def get_config(filename):
    file_pt = h5py.File(filename)
    arch_dict = json.loads(file_pt.attrs['model_config'].decode('utf-8'), encoding='utf-8')
    file_pt.close()
    return arch_dict['config']

def linear(x):
    return x
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def hard_sigmoid(x):
    tmp = x*0.2 + 0.5
    tmp[np.where(tmp<0)] = 0
    tmp[np.where(tmp>1)] = 1
    return tmp
    #return np.max(0, np.min(1, x*0.2 + 0.5))
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
        self.config = config['config']
        self.trainable_weights=[]
        
    def load_from_dataset(self, h5_dataset):
        for ind,name in enumerate(h5_dataset[self.name].attrs['weight_names']):
            self.trainable_weights[ind] = h5_dataset[self.name][name].value

class InputLayer(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        
class Embedding(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']
        self.dtype = self.config['dtype']
        self.kernel = None
        self.trainable_weights = [self.kernel]
    
    def load_weights(self, wgt=[]):
        self.kernel = wgt[0].value
        
    def __call__(self, input_vec):
        return self.kernel[input_vec]

class SimpleRNN(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        self.config = config['config']
        self.activation = activation_dict[self.config['activation']]
        self.return_sequences = self.config['return_sequences']
        self.units = self.config['units']
        self.use_bias = self.config['use_bias']
        self.kernel = None
        self.recurrent_kernel = None
        self.bias= None
        if self.use_bias:
            self.trainable_weight = [self.kernel, self.recurrent_kernel, self.bias]
        else:
            self.trainable_weight = [self.kernel, self.recurrent_kernel]
    
    def load_weights(self, wgt_list=[]):
        self.kernel = wgt_list[0].value
        self.recurrent_kernel = wgt_list[1].value
        if self.use_bias:
            self.bias = wgt_list[2].value
        else:
            self.bias = np.zeros(shape=(self.units,))
        
        
    def __call__(self, input_vec):
        timesteps = input_vec.shape[-2]
        if input_vec.ndim == 2:
            batch_size = 1
        else:
            batch_size = input_vec.shape[0]
        
        h_t  = np.zeros(shape=(batch_size, timesteps, self.units))
        x_w_b = np.dot(input_vec, self.kernel) + self.bias
        for tm in range(timesteps):
            h_t[:,tm,:] = self.activation( x_w_b[:,tm,:] + np.dot(h_t[:,tm-1,:], self.recurrent_kernel))
        if self.return_sequences:
            return h_t
        else:
            return h_t[:,-1,:]

class LSTM(SimpleRNN):
    def __init__(self, config={}):
        super().__init__(config)
        self.recurrent_activation = activation_dict[self.config['recurrent_activation']]
    
    def load_weights(self, wgt_list=[]):
        super().load_weights(wgt_list)

    def __call__(self, input_vec):
        timesteps = input_vec.shape[-2]
        if input_vec.ndim == 2:
            batch_size = 1
        else:
            batch_size = input_vec.shape[0]
        
        h_t = np.zeros(shape=(batch_size, timesteps, self.units))
        C_t = np.zeros(shape=(self.units,))
        x_w_b = np.dot(input_vec, self.kernel) + self.bias
        for tm in range(timesteps):
            mat_out = x_w_b[:,tm,:] + np.dot(h_t[:,tm-1,:], self.recurrent_kernel)
            i = self.recurrent_activation(mat_out[:, :self.units])
            f = self.recurrent_activation(mat_out[:, self.units: self.units * 2])
            C_t = f * C_t + i * self.activation(mat_out[:, self.units * 2: self.units * 3])
            o = self.recurrent_activation(mat_out[:, self.units * 3:])
            h_t[:,tm,:] = o * self.activation(C_t)
        
        if self.return_sequences:
            return h_t
        else:
            return h_t[:,-1,:]

class GRU(SimpleRNN):
    def __init__(self, config={}):
        super().__init__(config)
        self.recurrent_activation = activation_dict[self.config['recurrent_activation']]
        
    def load_weights(self, wgt_list=[]):
        super().load_weights(wgt_list)

    def __call__(self, input_vec):
        timesteps = input_vec.shape[-2]
        if input_vec.ndim == 2:
            batch_size = 1
        else:
            batch_size = input_vec.shape[0]
            
        h_t = np.zeros(shape=(batch_size, timesteps, self.units))
        h_bar = np.zeros(shape=(self.units,))
        x_w_b = np.dot(input_vec, self.kernel) + self.bias
        
        for tm in range(timesteps):
            mat_out = x_w_b[:,tm,:] + np.dot(h_t[:,tm-1,:], self.recurrent_kernel)
            z = self.recurrent_activation(mat_out[:,:self.units])
            r = self.recurrent_activation(mat_out[:,self.units: self.units * 2])
#             h_bar = self.activation(x_w_b[:,tm,self.units * 2:] + np.dot(r * h_t[:,tm-1,:], self.recurrent_kernel[:,self.units * 2:]))
            h_bar = self.activation(mat_out[:,self.units * 2:] - np.dot((1 - r) * h_t[:,tm-1,:], self.recurrent_kernel[:,self.units * 2:])) 
            h_t[:,tm,:] = z * h_t[:,tm-1,:] + (1 - z) * h_bar
        
        if self.return_sequences:
            return h_t
        else:
            return h_t[:,-1,:]

class Dense(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        self.activation = activation_dict[self.config['activation']]
        self.units = self.config['units']
        self.use_bias = self.config['use_bias']
        self.kernel = None
        self.bias= None
        if self.use_bias:
            self.trainable_weight = [self.kernel, self.bias]
        else:
            self.trainable_weight = [self.kernel]
    
    def load_weights(self, wgt_list=[]):
        self.kernel = wgt_list[0].value
        if self.use_bias:
            self.bias = wgt_list[1].value
        else:
            self.bias = np.zeros(shape=(self.units,))
    
    def __call__(self, input_vec):
        return self.activation(np.dot(input_vec, self.kernel) + self.bias)

class Add(Layer):
    def __init__(self, config={}):
        super().__init__(config)
            
    def __call__(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output

class Subtract(Layer):
    def __init__(self, config={}):
        super().__init__(config)
            
    def __call__(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')
        return inputs[0] - inputs[1]

class Multiply(Layer):
    def __init__(self, config={}):
        super().__init__(config)
            
    def __call__(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output *= inputs[i]
        return output

class Average(Layer):
    def __init__(self, config={}):
        super().__init__(config)
            
    def __call__(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output / len(inputs)

class Maximum(Layer):
    def __init__(self, config={}):
        super().__init__(config)
            
    def __call__(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = np.max(output, inputs[i])
        return output

class Minimum(Layer):
    def __init__(self, config={}):
        super().__init__(config)
            
    def __call__(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = np.min(output, inputs[i])
        return output

class Concatenate(Layer):
    def __init__(self, config={}):
        super().__init__(config)
        self.axis = self.config['axis']
            
    def __call__(self, inputs):
        return np.concatenate(inputs, axis=self.axis)


