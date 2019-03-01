import tensorflow as tf
import numpy as np

class LOGISTIC():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
       
        w_init = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.w = tf.get_variable('w', dtype=tf.float32,
                initializer = tf.cast(tf.constant(w_init), "float"))
        self.b = tf.get_variable('b', dtype=tf.float32,
                initializer = tf.zeros(output_dim))

        self.inp = tf.placeholder(tf.float32, [None, input_dim], 'inp')
        self.target = tf.placeholder(tf.float32, [None, output_dim], 'target')
        
        self.logits = self.inp @ self.w + self.b
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
        if optimizer == 'Adam':
            self.optim = tf.train.AdamOptimizer()
        self.step = self.optim.minimize(self.loss)
        
    def train_step(self, x, y):
        outputs = [self.step]
        return self.sess.run(outputs, feed_dict={self.inp:x, self.target:y})

    def forward(self, x, y=None):
        if len(x.shape) != 2:
            x = x.reshape(-1, np.prod(x.shape[1:]))
        # assert len(x.shape) == 2, x.shape
        assert x.shape[1] == self.input_dim, x.shape

        outputs = [self.pred]
        feed_dict = {self.inp:x}
        if y is not None:
            outputs += [self.loss, self.accuracy, self.confusion_matrix]
            feed_dict.update({self.target:y})
        return self.sess.run(outputs, feed_dict=feed_dict)
    
