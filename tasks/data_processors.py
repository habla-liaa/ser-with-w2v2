from paips.core import Task
import numpy as np
import pandas as pd
import copy

class PadDP(Task):
    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']
        col_out = self.parameters['column_out']
        max_length = self.parameters['max_length']
        axis = self.parameters.get('axis',0)
        
        def fn(x):
            if type(x).__name__ == 'GenericFile':
                x = x.load()
            shape_x = list(x.shape)
            shape_x[axis] = max_length
            out = np.zeros(shape_x)
            mask = np.zeros([max_length,])
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(0, min(x.shape[axis],max_length))
            out[slc] = x[slc]
            mask[:x.shape[axis]] = 1
            return out, mask

        y = list(zip(*map(fn,data[col_in])))
        out1 = pd.Series(y[0])
        out2 = pd.Series(y[1])
        out1.index = data.index
        out2.index = data.index
        data[col_out] = out1
        data['mask'] = out2
        
        return data

class ToNumpyDP(Task):
    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']

        return np.stack(data[col_in])

class SqueezeDP(Task):
    def process(self):
        data = self.parameters['in']
        col_in = self.parameters['column_in']
        col_out = self.parameters['column_out']

        data[col_out] = data[col_in].apply(lambda x: np.squeeze(x))

        return data

class LoadDataframeDP(Task):
    def process(self):
        col_in = self.parameters['column_in']
        col_out = self.parameters.get('column_out',col_in)
        data = self.parameters['in']
        exclude_cols = self.parameters.get('exclude_cols',None)
        include_cols = self.parameters.get('include_cols',None)
        order_by = self.parameters.get('order_by',None)
        
        def fn(x):
            if type(x).__name__ == 'GenericFile':
                x = x.load()
            if order_by is not None:
                x = x.sort_values(by = order_by)
            if exclude_cols is not None:
                x = x.drop(exclude_cols,axis=1)
            if include_cols is not None:
                x = x[include_cols]
            
            original_cols = list(x.columns)

            x = x.apply(lambda x: x.values,axis=1)
            x = np.stack(x.values)
            return x, original_cols

        y = list(zip(*map(fn,data[col_in])))
        out1 = pd.Series(y[0])
        out1.index = data.index
        out2 = pd.Series(y[1])
        out2.index = data.index
        data[col_out] = out1

        original_cols = out2.iloc[0]

        self.output_names = ['out', 'columns']
        return data, original_cols

class DownsampleDP(Task):
    def process(self):
        axis = self.parameters.get('axis',0)
        col_in = self.parameters['column_in']
        col_out = self.parameters.get('column_out',col_in)
        factor = self.parameters['factor']
        mode = self.parameters.get('mode','mean')

        data = self.parameters['in']

        def fn(x):
            target_axis_dim = x.shape[axis] + factor-(x.shape[axis]%factor)
            original_shape = list(x.shape)
            target_shape = copy.deepcopy(original_shape)
            target_shape[axis] = target_axis_dim

            y = np.zeros(target_shape)

            slc = [slice(None)] * len(target_shape)
            slc[axis] = slice(0, x.shape[axis])
            y[slc] = x
            y = np.swapaxes(y,axis,-1)
            reshape_shape = list(y.shape)
            reshape_shape[-1] = reshape_shape[-1]//factor
            reshape_shape.append(factor)
            y = np.reshape(y,reshape_shape)
            if mode == 'mean':
                y = np.mean(y,axis=-1)
            y = np.swapaxes(y,axis,-1)

            return y
            
        
        data[col_out] = data[col_in].apply(fn)
        return data

class OneHotVectorDP(Task):
    def process(self):
        data = self.parameters['in']
        column_in = self.parameters['column_in']
        column_out = self.parameters['column_out']
        mask = self.parameters.get('mask',None)
        frame_len = self.parameters.get('frame_len',None)
        n_classes = self.parameters['n_classes']

        def fn(x, mask):
            if frame_len:
                hotvector = np.zeros((frame_len,n_classes))
                hotvector[:,x] = 1
            else:
                hotvector = np.zeros((n_classes))
                hotvector[x] = 1
            if mask is not None:
                slice_mask = [slice(None)] * 2 + [0]*(mask.ndim-2)
                mask = mask[slice_mask]

                last_idx = np.max(np.argwhere(np.all(mask == 1,axis=1)))
                mask = np.ones((len(mask),n_classes))
                if last_idx + 1 < len(mask):
                    mask[last_idx+1:] = 0
                return hotvector*mask
            else:
                return hotvector

        if mask is not None:
            data[column_out] = data.apply(lambda x: fn(x[column_in],x[mask]),axis=1)
        else:
            data[column_out] = data[column_in].apply(lambda x: fn(x,None))

        return data

class NormalizeDP(Task):
    def process(self):
        data = self.parameters['in']
        statistics = self.parameters['statistics']
        column_in = self.parameters['column_in']
        column_out = self.parameters['column_out']

        if not isinstance(column_in, list):
            column_in = [column_in]
        if not isinstance(column_out, list):
            column_out = [column_out]

        columns = self.parameters.get('columns',None)

        if len(statistics.keys()) > 0:
            for col_in, col_out in zip(column_in,column_out):
                col_stats = statistics[col_in]
                group = list(col_stats.keys())[0]
                if group == 'global':
                    g_stats = col_stats[group]
                    if type(g_stats['mean']).__name__ == 'Series':      
                        if columns is not None:
                            g_stats['mean'] = g_stats['mean'].loc[columns]
                            g_stats['mean'] = g_stats['mean'].values
                    if type(g_stats['std']).__name__ == 'Series':
                        if columns is not None:
                            g_stats['std'] = g_stats['std'].loc[columns]
                        g_stats['std'] = g_stats['std'].values
                    data[col_out] = data.apply(lambda x: (x[col_in] - g_stats['mean'])/g_stats['std'],axis=1)
                else:
                    for g, g_stats in col_stats[group].items():
                        if type(g_stats['mean']).__name__ == 'Series':
                            if columns is not None:
                                g_stats['mean'] = g_stats['mean'].loc[columns]
                            g_stats['mean'] = g_stats['mean'].values
                        if type(g_stats['std']).__name__ == 'Series':
                            if columns is not None:
                                g_stats['std'] = g_stats['std'].loc[columns]
                            g_stats['std'] = g_stats['std'].values

                    data[col_out] = data.apply(lambda x: (x[col_in] - col_stats[group][x[group]]['mean'])/(col_stats[group][x[group]]['std']),axis=1)
        return data
