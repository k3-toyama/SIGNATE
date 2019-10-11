import tensorflow as tf


class NNLayer:
    def __init__(self, name, input_tensor, out_dim):
        self.name = name
        self.input = input_tensor
        self.out_dim = out_dim
        self.activation = 0
        self.weight = tf.Variable(initial_value=tf.truncated_normal(self.out_dim), name="")
        self.bias = None
        self.output = None


class NeuralNetwork:
    def __init__(self):
        return
