import math
import random

class Dataset(object):

    def __init__(self,instances,batch_size,shuffle):
        self.instances=instances
        self.batch_size=batch_size
        self.batch_count = math.ceil(len(instances) / batch_size)

        if shuffle:
            random.shuffle(self.instances)

    def get_batch(self,i):
        begin = i * self.batch_size
        batch_instances = self.instances[begin:begin + self.batch_size]

        return batch_instances