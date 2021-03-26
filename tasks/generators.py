from paips.core import Task, TaskIO
import numpy as np
from tensorflow.keras.utils import Sequence
import copy

class BatchGenerator_(Sequence):
    def __init__(self, data, shuffle=True, batch_task = None, batch_size=16, extra_data=None, batch_x=None, batch_y=None, seed=1234):
        np.random.seed(seed)
        self.data = data
        self.index = np.array(data.index)
        self.shuffle = shuffle
        self.batch_task = copy.deepcopy(batch_task)
        self.batch_size = batch_size
        self.batch_x = batch_x
        if not isinstance(self.batch_x,list):
            self.batch_x = [batch_x]
        self.batch_y = batch_y
        if not isinstance(self.batch_y,list):
            self.batch_y = [batch_y]
        if self.shuffle:
            self.index = np.random.permutation(self.index)

        if extra_data is not None:
            for k,v in extra_data.items():
                self.batch_task.parameters['in'][k] = TaskIO(v,'abcd',iotype='data',name='gen_extra',position=0)

    def __getitem__(self,idx):
        batch_idxs = np.arange(idx*self.batch_size,(idx+1)*self.batch_size)
        batch_idxs = np.take(self.index,batch_idxs,mode='wrap')
        batch_data = self.data.loc[batch_idxs]
        batch_data = TaskIO(batch_data,'abcd',iotype='data',name='batch_i',position=0)

        self.batch_task.parameters['in']['batch_data'] = batch_data

        outs = self.batch_task.run()
        outs = {k.split('->')[-1]: v for k,v in outs.items()}

        x = [outs[k].load() for k in self.batch_x]
        y = [outs[k].load() for k in self.batch_y]
        
        return x,y

    def on_epoch_end(self):
        if self.shuffle:
            self.index = np.random.permutation(self.index)
        
    def __len__(self):
        return int(len(self.data)//self.batch_size) + 1

class BatchGenerator(Task):
    def process(self):
        data = self.parameters['in']
        batch_task = copy.deepcopy(self.parameters['batch_task'])
        batch_x = self.parameters['x']
        batch_y = self.parameters['y']
        extra_data = self.parameters.get('extra_data',None)
        shuffle = self.parameters.get('shuffle',True)
        batch_size = self.parameters.get('batch_size',16)
        seed = self.parameters.get('seed',1234)

        return BatchGenerator_(data, shuffle, batch_task, batch_size, extra_data, batch_x, batch_y, seed)