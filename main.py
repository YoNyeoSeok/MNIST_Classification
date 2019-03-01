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

get_train_batch = get_batch(X_train, batch_size=100)

# model
if False:
    from model.tensorflow.logistic import LOGISTIC
    model = LOGISTIC(input_dim=feature, output_dim=n_class)
    fname = 'logs/tf_LOGISTIC_logs'
elif False:
    from model.tensorflow.feed_forward import FNN
    model = FNN(input_dim=feature, hidden_layers=[256, 256], output_dim=n_class)
    fname = 'logs/tf_FNN_logs'
else:
    from model.tensorflow.maxout import MaxoutNN
    model = MaxoutNN(input_dim=feature, hidden_layers=[50], output_dim=n_class)
    fname = 'logs/tf_MaxoutNN_logs'


# train, logs
model.train_setting(optimizer='Adam')
training_logs = {'loss':[[] for _ in range(32)], 'accuracy':[[] for _ in range(32)]}
validation_logs = {'loss':[[] for _ in range(32)], 'accuracy':[[] for _ in range(32)]}
test_logs = {'loss':[[] for _ in range(32)], 'accuracy':[[] for _ in range(32)]}
for j in range(32):
    model.model_init()
    for i in range(500+1):
        batch_idxs = get_train_batch.get_batch_idxs()
        model.train_step(x=X_train[batch_idxs], y=y_train_one_hot[batch_idxs])

        if i % 10 == 0:
            outputs = model.forward(x=X_train[batch_idxs], y=y_train_one_hot[batch_idxs])
            training_logs['loss'][j].append(outputs[1])
            training_logs['accuracy'][j].append(outputs[2])
            
        if i % 50 == 0:
            outputs = model.forward(x=X_test, y=y_test_one_hot)
            validation_logs['loss'][j].append(outputs[1])
            validation_logs['accuracy'][j].append(outputs[2])

    outputs = model.forward(x=X_test, y=y_test_one_hot)
    test_logs['loss'][j].append(outputs[1])
    test_logs['accuracy'][j].append(outputs[2])
    test_logs.update({'compusion_matrix': outputs[3]})
    if j % 4 == 0:
        print('test loss, test accuracy', outputs[1], outputs[2])
        print('compusion matrix:\n', outputs[3])

logs = {'training_logs':training_logs, 
        'validation_logs':validation_logs, 
        'test_logs':test_logs, 
        'validation_interval':5}

np.save(fname, logs)

