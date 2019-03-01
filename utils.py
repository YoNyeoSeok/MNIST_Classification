import numpy as np

class get_batch():
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size

        self.LENGTH = len(self.data)
        self.init()
        
    def get_batch_idxs(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        if self.loc + batch_size > self.LENGTH:
            self.init()
        self.loc += batch_size
        return self.idxs[self.loc - batch_size:self.loc]
        
    def get_batch_data(self, batch_size=None):
        idxs = self.get_batch_idxs(batch_size)
        return self.shuffled_data[self.loc:self.loc+len(idxs)]
        
    def init(self):
        self.reset()
        self.shuffle()

    def reset(self):
        self.loc=0

    def shuffle(self):
        self.idxs = np.random.permutation(self.LENGTH)
        self.shuffled_data = self.data[self.idxs]

