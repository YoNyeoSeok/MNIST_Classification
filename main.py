import numpy as np
from utils import *

# data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

shape = X_train[0].shape
feature = np.prod(shape)
X_train = X_train.reshape(-1, feature) / 255.
X_test = X_test.reshape(-1, feature) / 255.
n_class = len(np.unique(y_train))
y_train_one_hot = np.eye(n_class)[y_train]
y_test_one_hot = np.eye(n_class)[y_test]
assert X_train.shape[0] == y_train_one_hot.shape[0]
assert y_train_one_hot.shape[1] == n_class
assert len(y_train_one_hot.shape) == 2

get_train_batch = get_batch(X_train)

# model
import tensorflow as tf
tf_X = tf.placeholder(tf.float32, shape=[None, feature], name='X')
tf_y = tf.placeholder(tf.float32, shape=[None, n_class], name='y')

w_init = np.random.randn(feature, n_class).astype(np.float32) / np.sqrt(feature)
w = tf.get_variable("w", initializer = tf.constant(w_init))
b = tf.get_variable("b", initializer = tf.zeros(n_class))
y_pred = tf_X @ w + b

"""
from tensorflow.layers import BatchNormalization
norm = 'batch_norm'
activation = tf.sigmoid
n_hiddens = []
hidden = tf_X
for i, n_hidden in enumerate(n_hiddens):
    w = tf.get_variable("w"+str(i), initializer = tf.constant(w_init))
    b = tf.get_variable("b"+str(i), initializer = tf.zeros(n_class))
    if norm == 'Batch_norm':
        norm = BatchNormalization(name='batch_norm'+str(i))
        hidden = norm(hidden @ w + b)
    elif norm == 'None':
        hidden = hidden @ w + b
    hidden = activation(hidden) 
w = tf.get_variable("w"+str(len(n_hiddens)), initializer = tf.constant(w_init))
b = tf.get_variable("b"+str(len(n_hiddens)), initializer = tf.zeros(n_class))
if norm == 'Batch_norm':
    norm = BatchNormalization(name='batch_norm'+str(len(n_hiddens)))
    hidden = norm(hidden @ w + b)
elif norm == 'None':
    hidden = hidden @ w + b
y_pred = hidden
"""
#y_pred = tf_X @ w + b
#y_pred = norm(tf_X @ w + b)
confusion_matrix = tf.confusion_matrix(labels=tf.argmax(tf_y, 1), 
        predictions=tf.argmax(y_pred, 1))
 
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=y_pred)

opt = tf.train.AdamOptimizer(.001, .9, .999)
step = opt.minimize(loss)

# train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000+1):
        batch_idxs = get_train_batch.get_batch_idxs()
        sess.run([step], feed_dict={tf_X:X_train[batch_idxs], tf_y:y_train_one_hot[batch_idxs]})

        if i % 10 == 0:
            outputs = sess.run([y_pred, tf_y], 
                    feed_dict={tf_X:X_train[batch_idxs], tf_y:y_train_one_hot[batch_idxs]})
            assert np.argmax(outputs[0], 1).shape[0] == 32
            accuracy = np.mean(np.argmax(outputs[0], 1) == np.argmax(outputs[1], 1))
            print(i, accuracy)
    outputs = sess.run([y_pred, tf_y, confusion_matrix], 
            feed_dict={tf_X:X_test, tf_y:y_test_one_hot})
    accuracy = np.mean(np.argmax(outputs[0], 1) == np.argmax(outputs[1], 1))
    print('test accuracy: ', accuracy)

    print(outputs[2])
    


"""
batch_size = 32
def get_batch_idx(batch_size=batch_size, length=X_train.shape[0]):
    length = len(get_batch_idx.idxs)
    if get_batch_idx.loc + batch_size > length:
        get_batch_idx.loc = 0
        get_batch_idx.idxs = np.random.permutation(length)
    get_batch_idx.loc += batch_size
    return get_batch_idx.idxs[get_batch_idx.loc:get_batch_idx.loc+batch_size]
get_batch_idx.loc = 0
get_batch_idx.idxs = np.random.permutation(X_train.shape[0])
"""
