import numpy as np

logs = np.load('logs.npy').item()
print(logs.keys())

for k, v in logs.items():
    if type(v) is dict:
        for k_, v_ in v.items():
            print(k, k_, np.array(v_).shape)
    else:
        print(k, v)


for k, v in logs['test_logs'].items():
    print(k, v)
