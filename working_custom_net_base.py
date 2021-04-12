import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import math
import tensorflow as tf

# (ds_train, ds_test) = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, download=True)
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x = x.reshape((-1, 784))/255
x_test = x_test.reshape((-1, 784))/255


y = tf.one_hot(y, 10).numpy()
y_test = tf.one_hot(y_test, 10).numpy()


def relu(x):
    return tf.maximum(0, x)

def d_relu(x):
    return tf.where(x > 0, 1., 0.)

def tanh(x):
    e = math.e**x
    minus_e = 1/e
    return (e - minus_e) / (e + minus_e)


def sigmoid(x):
    return 1 / (1 + math.e**-x)

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig - sig**2

def linear(x):
    return x

def d_linear(x):
    return 1

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true)

def accuracy(y_true, y_pred):
    return np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)

d_dict = {sigmoid: d_sigmoid, linear: d_linear, relu: d_relu, mse: d_mse}

class Network:

    def __init__(self, n_in, cost, lr=3e-4):
        self.in_n = n_in
        self.layers = []
        self.output = None
        self.cost = cost
        self.lr = lr
        self.der_cost = d_dict[cost]


    def add_layer(self, layer):
        layer.lr = self.lr
        self.layers += [layer]

    def propagate(self, x):
        for layer in self.layers:
            x = layer(x)
        self.output = x
        return x

    def back_prop(self, x, label):
        self.propagate(x)
        print(f"accuracy: {np.mean(accuracy(label, self.output))}", f"loss: {np.mean(np.sum(self.cost(label, self.output), axis=-1))}")
        der = self.der_cost(label, self.output)
        for layer in reversed(self.layers):
            der = layer.back_prop(der)



class Layer:

    def __init__(self, n_out, activation=linear, starting_prune=1., beta=.0):
        self.n_in = None
        self.n_out = n_out
        self.weights = None
        self.biases = tf.zeros(n_out)
        self.activation = activation
        self.d_activation = d_dict[activation]
        self.starting_prune = starting_prune
        self.momentum = None
        self.active = None
        self.beta = beta
        self.x = None
        self.z = None
        self.a = None
        self.lr = None

    def init_weights(self):
        self.weights = tf.random.normal(shape=(self.n_in, self.n_out), stddev=(2/(self.n_in + self.n_out))**.5)


    def back_prop(self, der):

        d_out = tf.reshape(der * self.d_activation(self.z), (-1, 1, self.n_out))

        d_weights = d_out * tf.reshape(self.x, (-1, self.n_in, 1))



        self.weights -= self.lr * tf.reduce_mean(d_weights, axis=0)
        self.biases -= self.lr * tf.reduce_mean(d_out, axis=(0, 1))

        d_in = tf.reshape(self.weights, (1, self.n_in, self.n_out)) * d_out

        return tf.reduce_sum(d_in, axis=2)

    def __call__(self, *args, **kwargs):
        x = args[0]
        self.x = x
        if self.weights is None:
            self.n_in = x.shape[-1]
            self.init_weights()


        self.z = tf.matmul(x, self.weights) + self.biases
        self.a = self.activation(self.z)
        return self.a


network = Network(784, mse, lr=3e-3)
network.add_layer(Layer(200, relu))
network.add_layer(Layer(80, relu))
network.add_layer(Layer(10, sigmoid))

indices = np.arange(len(x))

batch_size = 256

for epoch in range(100):

    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        x_batch = tf.cast(tf.reshape(x_batch, (-1, 784)), dtype=tf.float32)
        y_batch = tf.cast(y_batch, tf.float32)

        network.back_prop(x_batch, y_batch)