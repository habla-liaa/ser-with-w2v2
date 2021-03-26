from paips.core import Task
import glob
from pathlib import Path
from pymediainfo import MediaInfo
import tqdm
import pandas as pd
import copy
import numpy as np

class Concatenate(Task):
	def process(self):
		return np.concatenate(self.parameters['in'],axis=self.parameters['axis'])

class Filter(Task):
    def process(self):
        exclude_values = self.parameters.get('exclude_values',None)
        include_values = self.parameters.get('include_values',None)
        column_filter = self.parameters.get('column',None)
        column_as = self.parameters.get('column_as',None)

        df_data = self.parameters['in']

        if column_as:
            df_data[column_filter] = df_data[column_filter].astype(column_as)

        if column_filter is None:
            raise Exception('Missing parameter: column')
        if exclude_values is not None:
            if not isinstance(exclude_values,list):
                exclude_values = [exclude_values]
            df_data = df_data[~df_data[column_filter].isin(exclude_values)]
        if include_values is not None:
            if not isinstance(include_values,list):
                include_values = [include_values]
            df_data = df_data[df_data[column_filter].isin(include_values)]

        return df_data

class Identity(Task):
    def process(self):
        return self.parameters['in']

class LabelEncoder(Task):
    def process(self):
        df_data = self.parameters['in']
        column = self.parameters['column']
        new_column = self.parameters.get('new_column','target')
        possible_labels = list(sorted(df_data[column].unique()))
        mapping = {j:i for i,j in enumerate(possible_labels)}
        df_data[new_column] = df_data[column].apply(lambda x: mapping[x])

        self.output_names = ['out','labels']

        return df_data, possible_labels

class Merge(Task):
    def process(self):
        in_tasks = self.parameters['in']
        if len(in_tasks)>1:
            df_merged = in_tasks[0]
            for df_i in in_tasks[1:]:
                cols = df_i.columns.difference(df_merged.columns)
                df_merged = pd.concat([df_merged,df_i[cols]],axis=1)
            return df_merged
        elif len(in_tasks)==1:
            return in_tasks[0]
        else:
            return None

class OutputMerger(Task):
    def process(self):
        outputs = []
        outputs_names = []
        for k, v in self.parameters['outputs'].items():
            outputs.append(v)
            outputs_names.append(k)
        
        self.output_names = outputs_names
        return tuple(outputs)

class Pool(Task):
    def process(self):
        pool_type = self.parameters.get('type','mean')
        axis = self.parameters.get('axis',-1)
        data = self.parameters.get('in',None)

        print('Input data shape: {}'.format(data.shape))
        if pool_type == 'mean':
            return np.mean(data,axis=axis)
        elif pool_type == 'sum':
            return np.sum(data,axis=axis)
        elif pool_type == 'argmax':
            return np.argmax(data,axis=axis)
        elif pool_type == 'max':
            return np.max(data,axis=axis)

class Relabel(Task):
    def process(self):
        enable = self.parameters.get('enable',True)
        data = self.parameters['in']
        relabels = self.parameters['relabels']

        if enable:
            for relabel in relabels:
                if 'column' in relabel:
                    if 'old_name' in relabel:
                        if not isinstance(relabel['old_name'],list):
                            relabel['old_name'] = [relabel['old_name']]
                        data.loc[data[relabel['column']].isin(relabel['old_name']),relabel['column']] = relabel['new_name']
                    elif 'mapping' in relabel:
                        data[relabel['column']] = data[relabel['column']].apply(lambda x: relabel['mapping'][x])
        return data