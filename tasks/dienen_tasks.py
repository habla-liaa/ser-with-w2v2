from paips.core import Task
import tqdm
import pandas as pd
import numpy as np
from deytah.batch_generator import BatchGenerator
from deytah.core import DataProcessor

from dienen import Model
from kahnfigh import Config
import os
import copy

import glob
from pathlib import Path
import joblib

from timeit import default_timer as timer

from swissknife.aws import S3File
from paips.utils import GenericFile

from IPython import embed

class DienenModel(Task):
    def process(self):
        if self.parameters.get('bukh_experiment',None):
            parents = ['/'.join(p.split('/')[:-1]) for p in self.parameters.find_path('bukh')]
            for parent in parents:
                parent = parent + '/bukh_experiment'
                self.parameters[parent] = self.parameters['bukh_experiment']
            self.parameters['bukh_experiment'].set_checkpoint_path(str(Path(self.global_parameters['cache_path'],self.task_hash,'checkpoints').absolute()))

        if self.parameters.get('wandb_run',None):
            parents = ['/'.join(p.split('/')[:-1]) for p in self.parameters.find_path('wandb_run')]
            for parent in parents:
                parent = parent + '/wandb_run'
                self.parameters[parent] = self.parameters['wandb_run']
            import wandb
            wandb.init(id = self.parameters['wandb_run'], resume=self.parameters['wandb_run'])

        if 'epochs' in self.parameters:
            self.parameters['dienen_config/Model/Training/n_epochs'] = self.parameters['epochs']

        dienen_model = Model(self.parameters['dienen_config'])

        seed = self.parameters.get('seed',1234)
        dienen_model.set_seed(seed)

        dienen_model.set_model_path(self.cache_dir)

        cache_dir = GenericFile(self.cache_dir,'checkpoints')
        ckpt_exportdir = GenericFile(self.export_path,'checkpoints')

        if not Path(ckpt_exportdir.local_filename).exists():
            if not Path(cache_dir.local_filename).exists():
                cache_dir.mkdir(parents=True,exist_ok=True)
            if not ckpt_exportdir.parent.exists():
                ckpt_exportdir.parent.mkdir(parents=True,exist_ok=True)

            if not Path(ckpt_exportdir.local_filename).exists() and not Path(ckpt_exportdir.local_filename).is_symlink():
                os.symlink(cache_dir.local_filename,ckpt_exportdir.local_filename)
                
        train_data = self.parameters['train_data']
        validation_data = self.parameters.get('validation_data',None)

        dienen_model.set_data(train_data, validation = validation_data)
        keras_model = dienen_model.build()

        #Resume training if checkpoints exist
        last_epoch = -1
        
        if self.cache:
            metadata_path = GenericFile(cache_dir,'metadata')
            if metadata_path.exists():
                metadata = metadata_path.load()
                last_epoch = max([epoch['step'] for epoch in metadata])
                metric_val_data = [[epoch['metric_val'] for epoch in metadata],[epoch['step'] for epoch in metadata]]
                best_epoch = metric_val_data[1][metric_val_data[0].index(min(metric_val_data[0]))]
                has_earlystop = Config(self.parameters['dienen_config']).find_keys('EarlyStopping')
                if len(has_earlystop)>0:
                    patience = Config(self.parameters['dienen_config'])[has_earlystop[0]].get('patience',1)
                    if last_epoch - best_epoch >= patience:
                        last_epoch = self.parameters['dienen_config/Model/Training/n_epochs']
                        weights_path = [epoch['weights_path'] for epoch in metadata if epoch['step'] == best_epoch][0]
                        opt_weights_path = [epoch['opt_weights_path'] for epoch in metadata if epoch['step'] == best_epoch][0]
                    else:
                        weights_path = [epoch['weights_path'] for epoch in metadata if epoch['step'] == best_epoch][0]
                        opt_weights_path = [epoch['opt_weights_path'] for epoch in metadata if epoch['step'] == best_epoch][0]            
                else:
                    weights_path = [epoch['weights_path'] for epoch in metadata if epoch['step'] == last_epoch][0]
                    opt_weights_path = [epoch['opt_weights_path'] for epoch in metadata if epoch['step'] == last_epoch][0]

                self.cache_dir = GenericFile(self.cache_dir)
                if not Path(weights_path).exists() and self.cache_dir.filesystem == 's3':
                    s3_wpath = S3File(str(self.cache_dir),'checkpoints',Path(weights_path).name)
                    if s3_wpath.exists():
                        s3_wpath.download(Path(weights_path))

                if not Path(opt_weights_path).exists() and self.cache_dir.filesystem == 's3':
                    s3_opath = S3File(str(self.cache_dir),'checkpoints',Path(opt_weights_path).name)
                    if s3_opath.exists():
                        s3_opath.download(Path(opt_weights_path))
                
                if Path(weights_path).exists():
                    dienen_model.set_weights(weights_path)
                if Path(opt_weights_path).exists():
                    dienen_model.set_optimizer_weights(opt_weights_path)

        if 'extra_data' in self.parameters:
            dienen_model.set_extra_data(self.parameters['extra_data'])

        dienen_model.cache = self.cache
        if last_epoch < self.parameters['dienen_config/Model/Training/n_epochs']:
            dienen_model.fit(train_data, validation_data = validation_data, from_epoch=last_epoch+1)

        dienen_model.load_weights(strategy='min')
        dienen_model.clear_session()

        return dienen_model

    def make_hash_dict(self):
        from paips.utils.settings import symbols
        
        self.hash_dict = copy.deepcopy(self.parameters)
        #Remove not cacheable parameters
        if not isinstance(self.hash_dict, Config):
            self.hash_dict = Config(self.hash_dict)
        if not isinstance(self.parameters, Config):
            self.parameters = Config(self.parameters)

        epochs_path = ['dienen_config/Model/Training/n_epochs', 'epochs']

        for epoch_path in epochs_path:
            if epoch_path in self.hash_dict:
                if not epoch_path.startswith('!nocache'):
                    self.hash_dict[epoch_path] = '!nocache {}'.format(self.hash_dict[epoch_path])

        _ = self.hash_dict.find_path(symbols['nocache'],mode='startswith',action='remove_value')
        _ = self.parameters.find_path(symbols['nocache'],mode='startswith',action='remove_substring')

class DienenPredict(Task):
    def process(self):
        model = self.parameters['model']
        data = self.parameters['data']
        return_targets = self.parameters.get('return_targets',True)
        deytah_process = self.parameters.get('deytah_process',None)
        deytah_keys = self.parameters.get('deytah_keys',None)
        activations = self.parameters.get('activations','output')

        if deytah_process is not None:
            deytah_process = DataProcessor(deytah_process)
        group_predictions = self.parameters.get('group_predictions_by',None)
        return_as_metadata = self.parameters.get('return_as_metadata',False)
        batch_as_time = self.parameters.get('batch_as_axis',None)
        return_column = self.parameters.get('return_column',None)

        if return_as_metadata:
            metadata = []
            prediction_dir = Path(self.cache_dir,'predictions')
            if not prediction_dir.exists():
                prediction_dir.mkdir(parents=True,exist_ok = True)

            if group_predictions:
                groups = data[group_predictions].unique()
                for group in tqdm.tqdm(groups):
                    file_dir = Path(prediction_dir, '{}'.format(group))
                    if self.cache and file_dir.exists():
                        print('Caching {}'.format(group))
                    else:
                        data_i = data[data[group_predictions] == group]
                        start = timer()
                        if deytah_process != None:
                            data_i = deytah_process.process(data_i.to_dict('list'))
                            if deytah_keys != None:
                                data_i = [data_i[k] for k in ['audios','masks']]
                            else:
                                data_i = data_i.values()
                        end = timer()
                        print('Deytah processing: {}'.format(start-end))
                        start = timer()
                        prediction_i = model.predict(data_i,output=activations)
                        if activations != 'output':
                            prediction_i = [prediction_i[k] for k in activations]
                        else:
                            prediction_i = prediction_i.values()
                        prediction_i = np.array(prediction_i)
                        perm = list(range(prediction_i.ndim))
                        perm[0] = 1
                        perm[1] = 0
                        prediction_i = np.transpose(prediction_i,perm)
                        if batch_as_time:
                            prediction_i = np.concatenate(prediction_i,axis=batch_as_time)
                        end = timer()
                        print('Prediction: {}'.format(start-end))
                        start = timer()
                        joblib.dump(prediction_i,file_dir,compress=3)
                        end = timer()
                        print('Saving: {}'.format(start-end))
                        #np.save(file_dir,prediction_i)
                    metadata_row = data[data[group_predictions] == group].groupby('file_name').agg(np.unique)
                    metadata_row = metadata_row.reset_index()
                    metadata_row['embedding_filename'] = str(file_dir.absolute())
                    metadata.append(metadata_row)

            return pd.concat(metadata).reset_index().drop('index',axis=1)

        else:
            if isinstance(data,tuple) or isinstance(data,list):
                if not isinstance(data[0],list):
                    generator_samples = len(data[0])
                else:
                    generator_samples = len(data[0][0])
                predictions = model.predict(data[0])[:generator_samples]
            else:           
                generator_samples = len(data.index)
                predictions = model.predict(data)[:generator_samples]
            return_data = (predictions,)
            self.output_names = ['predictions']
            if return_targets:
                if isinstance(data,tuple) or isinstance(data,list):
                    targets = data[1]
                else:
                    check_shape = data.__getitem__(0)[1]
                    if isinstance(check_shape,list):
                        targets = np.array([data.__getitem__(i)[1][0] for i in range(len(data))])
                    else:
                        targets = np.array([data.__getitem__(i)[1] for i in range(len(data))])
                    targets = np.reshape(targets,(-1,)+targets.shape[2:])
                    targets = targets[:generator_samples]

                assert len(targets) == len(predictions)
                self.output_names += ['targets']
                return_data += (targets,)
            
            if return_column:
                return_data += (data.data[return_column],)
                self.output_names += (return_column,)


            return return_data