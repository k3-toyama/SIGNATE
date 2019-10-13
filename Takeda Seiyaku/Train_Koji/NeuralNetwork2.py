import tensorflow as tf


class NN_Activation:
    LINEAR = 0
    SIGMOID = 1
    RELU = 2
    TANH = 3

    def activate(self, f, x, name=""):
        if f == NN_Activation.SIGMOID:
            out = tf.nn.sigmoid(x, name)
        elif f == NN_Activation.RELU:
            out = tf.nn.relu(x, name)
        elif f == NN_Activation.TANH:
            out = tf.nn.tanh(x, name)
        else:
            out = x

        return out


class NN_Layer:
    """
    ニューラルネットワークのひとつの層
    入力に重み、バイアス、活性化関数をかけて出力する
    """
    def __init__(self, name, input_tensor, in_dim, out_dim=1):
        """
        :param name: 層の名前
        :param input_tensor:層の入力になるテンソル
        :param in_dim: 層の入力テンソルの次元数
        :param out_dim: 層の出力テンソルの次元数
        """
        self.name = name
        self.input = input_tensor
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = None
        self.bias = None
        self.activation_function = None
        self.output = None

    @property
    def w_dim(self):
        """重みテンソルの次元、初期値の設定の際使うと便利"""
        return [self.in_dim, self.out_dim]

    @property
    def b_dim(self):
        """バイアステンソルの次元、初期値の設定の際使うと便利"""
        return self.out_dim

    def init_w(self, w_init=None):
        if w_init is None:
            w_init = tf.truncated_normal(self.w_dim)
        self.weight = tf.Variable(initial_value=w_init, name="w_"+self.name)

    def init_b(self, b_init=None):
        if b_init is None:
            self.bias = tf.constant(tf.zeros(self.b_dim), name="b_"+self.name)
        else:
            self.bias = tf.Variable(initial_value=b_init, name="b_"+self.name)

    def set_activation(self, activation):
        self.activation_function = activation

    def set_param(self, w_init=None, b_init=None, activation=NN_Activation.LINEAR):
        """
        全バラメータを設定する
        :param w_init: 重みの初期値、入力しなければ正規分布の乱数
        :param b_init: バイアスの初期値、入力しなければゼロベクトルの定数（バイアスなし）
        :param activation: 活性化関数の種類、NN_Activationに設定されている定数で指定、入力しなければ活性化関数なし
        """
        self.init_w(w_init)
        self.init_b(b_init)
        self.set_activation(activation)

    def output(self, drop=0.5):
        """
        設定をもとに出力テンソルを組む
        :param drop:
        :return: 層の出力テンソル
        """
        affine = tf.matmul(self.input, self.weight) + self.bias
        self.output = NN_Activation.activate(self.activation_function, affine)
        return self.output


class NeuralNetwork:
    def __init__(self):
        return
