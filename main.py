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

confusion_matrix = tf.confusion_matrix(labels=tf.argmax(tf_y, 1), 
        predictions=tf.argmax(y_pred, 1))
 
losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=y_pred)
loss = tf.reduce_mean(losses)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(tf_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

opt = tf.train.AdamOptimizer(.001, .9, .999)
step = opt.minimize(loss)

# train, log
training_logs = {'loss':[[] for _ in range(5)], 'accuracy':[[] for _ in range(5)]}
validation_logs = {'loss':[[] for _ in range(5)], 'accuracy':[[] for _ in range(5)]}
test_logs = {'loss':[[] for _ in range(5)], 'accuracy':[[] for _ in range(5)]}
for j in range(5):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(500+1):
            batch_idxs = get_train_batch.get_batch_idxs()
            outputs = sess.run([loss, accuracy, step], 
                    feed_dict={tf_X:X_train[batch_idxs], tf_y:y_train_one_hot[batch_idxs]})
            training_logs['loss'][j].append(outputs[0])
            training_logs['accuracy'][j].append(outputs[1])
            
            if i % 10 == 0:
                outputs = sess.run([loss, accuracy], 
                        feed_dict={tf_X:X_test, tf_y:y_test_one_hot})
                validation_logs['loss'][j].append(outputs[0])
                validation_logs['accuracy'][j].append(outputs[1])
    
        outputs = sess.run([loss, accuracy, confusion_matrix],
            feed_dict={tf_X:X_test, tf_y:y_test_one_hot})
        test_logs['loss'][j].append(outputs[0])
        test_logs['accuracy'][j].append(outputs[1])
        test_logs.update({'compusion_matrix': outputs[2]})
        print('compusion_matrix:\n', outputs[2])

logs = {'training_logs':training_logs, 'validation_logs':validation_logs, 'test_logs':test_logs, 'validation_interval':10}
np.save('logs', logs)

# model save
