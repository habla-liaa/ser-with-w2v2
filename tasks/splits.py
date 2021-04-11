from paips.core import Task
import glob
from pathlib import Path
import tqdm
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import copy
import numpy as np

class GroupSplit(Task):
    def process(self):
        df_original = self.parameters['in']
        df = copy.deepcopy(df_original)
        splits = self.parameters['splits']
        group_column = self.parameters.get('group_column',None)
        out_column = self.parameters.get('column','partition')

        seed = self.parameters.get('seed',1234)
        np.random.seed(seed)

        groups = df[group_column].unique()
        non_random_splits = [k for k,v in splits.items() if not isinstance(v,float)]
        random_splits = [k for k,v in splits.items() if isinstance(v,float)]
        for k in non_random_splits:
            v = splits[k]
            if not isinstance(v,list):
                v = [v]
            splits[k] = list(df[df[group_column].isin(v)].index)

        assigned_indexs = [idx for k,v in splits.items() if isinstance(v,list) for idx in v]
        unassigned_df = df[~df.index.isin(assigned_indexs)]
        unassigned_groups = unassigned_df[group_column].unique()

        n_sample = len(unassigned_groups)

        if len(random_splits)>0:
            for k in random_splits[:-1]:
                n = int(splits[k]*n_sample)
                sampled_groups = np.random.choice(unassigned_groups,size=n,replace=False)
                df_sampled = unassigned_df.loc[unassigned_df[group_column].isin(sampled_groups)]
                splits[k] = df_sampled.index
                assigned_indexs.extend(list(df_sampled.index))
                unassigned_df = unassigned_df.drop(df_sampled.index)
                unassigned_groups = list(set(unassigned_groups) - set(sampled_groups))
            splits[random_splits[-1]] = unassigned_df.index
      
        for k,v in splits.items():
            df.loc[v,out_column] = k
            
        return df

class LeaveOneOut(Task):
    def process(self):
        groups = self.parameters['in'][self.parameters['group_col']].unique()
        self.output_names = ['train','test']

        test_folds = [[group] for group in groups]
        train_folds = [[f for f in groups if f not in test_fold] for test_fold in test_folds]
        
        return train_folds, test_folds
        
class RandomSplit(Task):
    def process(self):
        df_original = self.parameters['in']
        df = copy.deepcopy(df_original)
        df_filter = self.parameters.get('filter',None)
        df_col = self.parameters.get('column',None)
        seed = self.parameters.get('seed',1234)
        if df_filter:
            df = df.loc[df[df_col]==df_filter]
        splits = self.parameters.get('splits',None)
        data_len = len(df)
        stratified = self.parameters.get('stratified',None)
        if stratified is not None:
            stratified_classes = df[stratified].value_counts()
	
	if isinstance(splits,int):
            splits = {'fold_{}'.format(i): 1.0/splits for i in range(splits)}
        
        for i, (s_name, s_prop) in enumerate(splits.items()):
            if i<len(splits)-1:
                if stratified:
                    out_i = []
                    for strata_name,strata_n in stratified_classes.items():
                        strata_data = df.loc[df[stratified]==strata_name]
                        out_i.append(strata_data.sample(int(s_prop*strata_n),random_state=seed))
                    out_i = pd.concat(out_i)
                else:
                    n = int(s_prop*data_len)
                    out_i = df.sample(n=n, random_state=seed)

                sampled_idxs = out_i.index
                df_original.loc[sampled_idxs,df_col] = s_name
                df = df.loc[~df.index.isin(sampled_idxs)]
            else:
                df_original.loc[df.index,df_col] = s_name

        return df_original

class Split(Task):
	def process(self):
		data = self.parameters['in']
		seed = self.parameters.get('seed',1234)
		#Partitions are already provided in a column:
		split_col = self.parameters.get('split_col',None)
		if split_col:
			partition_names = data[split_col].unique()
			partition_names.sort()
			group_outputs = self.parameters.get('group_outputs',None)
			if group_outputs:
				output_groups = []
				self.output_names = []
				for group_name, group_items in group_outputs.items():
					output_group_i = data[data[split_col].isin(group_items)]
					output_groups.append(output_group_i)
					self.output_names.append(group_name)
				return tuple(output_groups)
			else:
				self.output_names = partition_names
				return tuple([data[data[split_col] == part_name] for part_name in partition_names])
		
		ratios = self.parameters.get('ratios',None)

		if ratios:
			stratified = self.parameters.get('stratified',None)
			
			out = []
			self.output_names = []

			if stratified is not None:
				stratified_classes = data[stratified].value_counts()

			data_len = len(data)

			for i, (k, v) in enumerate(ratios.items()):
				if i<len(ratios)-1:
					if stratified is not None:
						out_i = []
						for strata_name,strata_n in stratified_classes.items():
							strata_data = data.loc[data[stratified]==strata_name]
							out_i.append(strata_data.sample(int(v*strata_n),random_state=seed))
						try:
							out_i = pd.concat(out_i)
						except:
							from IPython import embed
							embed()
					else:
						n = int(v*data_len)
						out_i = data.sample(n=n, random_state=seed)
					sampled_idxs = out_i.index
					out.append(out_i)
					data = data.loc[~data.index.isin(sampled_idxs)]
				else:
					out.append(data)
				self.output_names.append(k)
			
			if stratified:
				for data_i,part_name in zip(out,self.output_names):
					print(part_name)
					print('------------------------------------------\n')
					print(data[stratified].value_counts(normalize=True))
					print(len(data_i))
					print('')

			return tuple(out)

		#Make partitions using groups from a column (ie speakers) as samples (REVISAR)
		group = self.parameters.get('group',None)
		seed = self.parameters.get('seed',1234)
		if group:
			test_size = self.parameters.get('test_size',0.2)
			train_size = self.parameters.get('train_size',None)
			gss = GroupShuffleSplit(n_splits=1,test_size=test_size,train_size=train_size,random_state=seed)
			split_idxs = list(gss.split(data.values,groups=data[group].values))
			self.output_names = ['train','test']
			return (data.iloc[split_idxs[0][0]],data.iloc[split_idxs[0][1]])

		lists = self.parameters.get('lists',None)
		list_field = self.parameters.get('list_item_column',None)
		if lists is not None:
			outs = []
			output_names = []
			for k,v in lists.items():
				output_names.append(k)
				with open(str(Path(v).expanduser()),'r') as f:
					list_items = f.read().splitlines()
				if not list_field == 'use_index':
					index_name = data.index.name
					data.reset_index(inplace=True)
					data = data.set_index(list_field)

				list_items = [x for x in list_items if x in data.index]

				df_i = data.loc[list_items]

				if not list_field == 'use_index':
					df_i = df_i.set_index(index_name)

				outs.append(df_i)

			self.output_names = output_names
			
			return tuple(outs)
