import tensorflow as tf
import numpy as np

class MaxoutNN():
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim

        self.inp = tf.placeholder(tf.float32, [None, self.input_dim], 'inp')
        self.target = tf.placeholder(tf.float32, [None, self.output_dim], 'target')
        self.train = tf.placeholder(tf.bool, [], 'train')

        self.construct_model()
        self.sess = tf.Session()
        self.optim = None

    def construct_model(self, initializer='glorot_normal', 
            norm='None', activation='maxout2'):
        self.initializer = get_initializer(initializer)

        if 'maxout' not in activation:
            self.activation = get_activation(activation)
            self.normalize = get_normalize(norm, self.activation, self.train)
        
        if 'maxout' in activation:
            self.maxout_hidden_layers = self.hidden_layers
            self.hidden_layers = [h*int(activation.lstrip('maxout')) for h in self.hidden_layers]

        self.ws = [tf.get_variable('w0', 
            shape = [self.input_dim, self.hidden_layers[0]],
            dtype = tf.float32, initializer = self.initializer)]
        self.bs = [tf.get_variable('b0', 
            shape = [self.hidden_layers[0]],
            dtype = tf.float32, initializer = tf.initializers.zeros)]
        if 'maxout' in activation:
            self.activation = get_activation(activation, self.maxout_hidden_layers[0])
            self.normalize = get_normalize(norm, self.activation, self.train)
            self.hiddens = [self.normalize(self.inp @ self.ws[0] + self.bs[0])]
        else:
            self.hiddens = [self.normalize(self.inp @ self.ws[0] + self.bs[0])]

        for i in range(1, len(self.hidden_layers)):
            self.ws += [tf.get_variable('w'+str(i), 
                shape = [self.hidden_layers[i-1] if 'maxout' not in activation else
                        self.maxout_hidden_layers[i-1], self.hidden_layers[i]],
                dtype = tf.float32, initializer = self.initializer)]
            self.bs += [tf.get_variable('b'+str(i), 
                shape = [self.hidden_layers[i]],
                dtype = tf.float32, initializer = tf.initializers.zeros)]
            if 'maxout' in activation:
                self.activation = get_activation(activation, self.maxout_hidden_layers[i])
                self.normalize = get_normalize(norm, self.activation, self.train)
                self.hiddens.append(self.normalize(self.hiddens[-1] @ self.ws[i] + self.bs[i]))
            else:
                self.hiddens.append(self.normalize(self.hiddens[-1] @ self.ws[i] + self.bs[i]))

        self.ws += [tf.get_variable('w'+str(len(self.hidden_layers)), 
            shape = [self.hidden_layers[-1] if 'maxout' not in activation else
                    self.maxout_hidden_layers[-1], self.output_dim],
            dtype = tf.float32, initializer = self.initializer)]
        self.bs += [tf.get_variable('b'+str(len(self.hidden_layers)), 
            shape = [self.output_dim],
            dtype = tf.float32, initializer = tf.initializers.zeros)]
        self.logits = self.hiddens[-1] @ self.ws[len(self.hidden_layers)] + self.bs[len(self.hidden_layers)]
        self.pred = tf.nn.softmax(self.logits, 1)

        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target,
                    logits = self.logits))

        self.correct_pred = tf.equal(
                tf.argmax(self.pred, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_pred, "float"))

        self.confusion_matrix = tf.confusion_matrix(labels = tf.argmax(self.target, 1),
                predictions = tf.argmax(self.pred, 1))

        self.sess = tf.Session()
        self.optim = None
   
    def model_init(self):
        self.sess.run(tf.global_variables_initializer())

    def train_setting(self, optimizer='Adam'):
        self.optim = get_optimizer(optimizer)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.step = self.optim.minimize(self.loss)

    def train_step(self, x, y):
        outputs = [self.step]
        return self.sess.run(outputs, feed_dict = {self.inp:x, self.target:y, self.train:True})

    def forward(self, x, y=None, train=False):
        if len(x.shape) != 2:
            x = x.reshape(-1, np.prod(x.shape[1:]))
        # assert len(x.shape) == 2, x.shape
        assert x.shape[1] == self.input_dim, x.shape

        outputs = [self.pred]
        feed_dict = {self.inp:x, self.train:train}
        if y is not None:
            outputs += [self.loss, self.accuracy, self.confusion_matrix]
            feed_dict.update({self.target:y})
        return self.sess.run(outputs, feed_dict=feed_dict)

def get_activation(activation, param=1):
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'sigmoid':
        return tf.nn.sigmoid
    elif activation == 'tanh':
        return tf.nn.tanh
    elif 'maxout' in activation:
        def activation(inp):
            return tf.contrib.layers.maxout(inp, param)
        return activation
    else:
        assert False, activation
       
def get_normalize(norm, activation, train):
    if norm == 'batch_norm':
        def normalize(inp):
            return tf.contrib.layers.batch_norm(inp, 
                    activation_fn = activation)
    elif norm == 'layer_norm':
        def normalize(inp):
            return tf.contrib.layers.layer_norm(inp, 
                    activation_fn = activation)
    elif 'dropout' in norm:
        def normalize(inp):
            return tf.contrib.layers.dropout(activation(inp),
                    keep_prob=1-float(norm.lstrip('dropout')),
                    is_training=train)
    elif norm == 'None':
        def normalize(inp):
            return activation(inp)
    else:
        assert False, norm
    return normalize 

def get_initializer(initializer):
    if initializer == 'truncated_normal':
        return tf.initializers.truncated_normal
    elif initializer == 'glorot_normal':
        return tf.initializers.glorot_normal
    elif initializer == 'glorot_uniform':
        return tf.initializers.glorot_uniform
    elif initializer == 'he_normal':
        return tf.initializers.hi_normal
    elif initializer == 'he_uniform':
        return tf.initializers.hi_uniform
    elif initializer == 'lecun_normal':
        return tf.initializers.lecun_normal
    elif initializer == 'lecun_uniform':
        return tf.initializers.lecun_uniform
    else:
        assert False, initializer

def get_optimizer(optimizer):
    if optimizer == 'Adam':
        return tf.train.AdamOptimizer()
