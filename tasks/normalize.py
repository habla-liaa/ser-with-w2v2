from paips.core import Task
import tqdm
import pandas as pd
import numpy as np

from IPython import embed

class BatchedMVN:
    def __init__(self, axis=0):
        self.moving_mean = 0
        self.moving_var = 0
        self.data_size = 0
        self.axis = axis

    def update(self,data):
        if self.axis is None:
            n = data.size
        else:
            n = data.shape[self.axis]
        batch_mean = np.mean(data,axis=self.axis)
        batch_std = np.std(data,axis=self.axis)

        self.moving_mean = (self.data_size*self.moving_mean + n*batch_mean)/(n+self.data_size)

        self.moving_var = (n*batch_std**2)/(n+self.data_size) + (self.data_size*self.moving_var)/(n+self.data_size) +\
        (n*self.data_size*(self.moving_mean - batch_mean)**2)/((n+self.data_size)**2)
        #if np.any(np.isnan(self.moving_var)):
        #    embed()
        self.data_size += n

    def get_mean_and_std(self):
        return self.moving_mean, self.moving_var**0.5

class NormalizationStatistics(Task):
    def process(self):
        data = self.parameters['in']
        normalization_by = self.parameters.get('by','global')
        column = self.parameters.get('column',None)
        if not isinstance(column,list):
            column = [column]
        mode = self.parameters.get('mode','mvn')
        axis = self.parameters.get('axis',0)

        if normalization_by == 'global':
            statistics = {}
            for col in column:
                feat_i_stats = {'global': {}}
                idxs = data.index
                if mode == 'mvn':
                    batched_mvn = BatchedMVN(axis=axis)
                    for idx in idxs:
                        data_i = data.loc[idx][col]
                        data_i_type = type(data_i).__name__
                        if (data_i_type == 'GenericFile'):
                            data_i = data_i.load()
                        elif (data_i_type == 'PosixPath') or (data_i_type == 'str'):
                            data_i = joblib.load(data_i)
                        else:
                            pass
                        batched_mvn.update(data_i)

                    mean, std = batched_mvn.get_mean_and_std()
                    feat_i_stats['global'] = dict(mean = mean, std = std)
                    statistics[col] = feat_i_stats
            return statistics

        elif normalization_by is None:
            return {}
        else:
            statistics = {}
            groups = data[normalization_by].unique()
            for col in column:
                feat_i_stats = {normalization_by:{}}
                for g in groups:
                    grouped_data = data.loc[data[normalization_by] == g][col]
                    group_idxs = grouped_data.index
                    if mode == 'mvn':
                        group_mvn = BatchedMVN(axis=axis)
                        for idx in group_idxs:
                            data_i = grouped_data.loc[idx]
                            data_i_type = type(data_i).__name__
                            if (data_i_type == 'GenericFile'):
                                data_i = data_i.load()
                            elif (data_i_type == 'PosixPath') or (data_i_type == 'str'):
                                data_i = joblib.load(data_i)
                            else:
                                pass
                            group_mvn.update(data_i)

                        mean, std = group_mvn.get_mean_and_std()
                        
                        feat_i_stats[normalization_by][g] = dict(mean = mean, std = std)
                statistics[col] = feat_i_stats

            return statistics
                    

                

                
        