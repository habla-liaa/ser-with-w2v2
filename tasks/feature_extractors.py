import fairseq
import torch
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2CtcConfig, Wav2VecCtc
import librosa
from paips.core import Task
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib
from paips.utils import GenericFile
import opensmile

class OpensmileExtractor(Task):
    def process(self):
        config = self.parameters['config']
        feature_level = self.parameters.get('feature_level','lld') #lld or func
        data = self.parameters['in']
        output_column = self.parameters['output_column']
        save_feature_files = self.parameters.get('save_feature_files',True)
        max_size = self.parameters.get('max_size',None)

        if save_feature_files:
            feature_output_path = GenericFile(self.global_parameters['cache_path'],self.task_hash,'features')
            feature_output_path.mkdir(parents=True)

        if config in opensmile.FeatureSet.__members__:
            feature_set = opensmile.FeatureSet[config]
        else:
            feature_set = config

        smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=feature_level,
        )

        def extract_embedding(x):
            output_feature_filename = GenericFile(feature_output_path,'{}.feat'.format(Path(x).stem))
            if output_feature_filename.exists() and self.cache:
                if save_feature_files:
                    return output_feature_filename
                else:
                    return output_feature_filename.load()
            else:
                try:
                    y = smile.process_file(x,channel=0)
                except:
                    embed()
                if max_size:
                    y = y[:max_size]
                if save_feature_files:
                    joblib.dump(y, output_feature_filename.local_filename)
                    output_feature_filename.upload_from(output_feature_filename.local_filename)
                    return output_feature_filename
                else:
                    return y
        
        tqdm.pandas()

        data[output_column] = data['filename'].progress_apply(extract_embedding)
        return data

class Spectrogram(Task):
    def process(self):
        spec_type = self.parameters.get('type','magnitude')
        frame_size = self.parameters['frame_size']
        hop_size = self.parameters.get('hop_size',frame_size//4)
        nfft = self.parameters.get('nfft',frame_size)
        window = self.parameters.get('window','hann')
        data = self.parameters['in']
        output_column = self.parameters['output_column']
        save_feature_files = self.parameters.get('save_feature_files',True)
        log_offset = self.parameters.get('log_offset',1e-12)

        if save_feature_files:
            feature_output_path = GenericFile(self.global_parameters['cache_path'],self.task_hash,'features')
            feature_output_path.mkdir(parents=True)

        def extract_embedding(x):
            output_feature_filename = GenericFile(feature_output_path,'{}.feat'.format(Path(x).stem))
            if output_feature_filename.exists() and self.cache:
                if save_feature_files:
                    return output_feature_filename
                else:
                    return output_feature_filename.load()
            else:
                x, fs = librosa.core.load(x,sr=None)
                X = librosa.stft(x, n_fft=nfft, hop_length=hop_size, win_length=frame_size, window=window)
                X = X.T
                if spec_type == 'magnitude':
                    y = np.abs(X)
                elif spec_type == 'complex':
                    y = X
                elif spec_type == 'log_magnitude':
                    y = np.log(np.abs(X)+log_offset)

                if save_feature_files:
                    joblib.dump(y, output_feature_filename.local_filename)
                    output_feature_filename.upload_from(output_feature_filename.local_filename)
                    return output_feature_filename
                else:
                    return y

        tqdm.pandas()

        data[output_column] = data['filename'].progress_apply(extract_embedding)
        return data

class Wav2Vec2Embeddings(Task):
    def build_wav2vec_model(self, model_path, dict_path=None):
        
        arg_override = {'activation_dropout': 0.0,
                        'attention_dropout': 0.0,
                        'dropout': 0.0,
                        'dropout_features': 0.0,
                        'dropout_input': 0.0,
                        'encoder_layerdrop': 0.0,
                        'pooler_dropout': 0.0}

        if dict_path:
            arg_override.update({"data": dict_path})

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], arg_overrides=arg_override)
        
        return model[0], cfg, task

    def process(self):
        data = self.parameters['in']
        model_path = str(Path(self.parameters['model_path']).expanduser())
        dict_path = self.parameters.get('dict_path',None)
        if dict_path is not None:
            dict_path = str(Path(dict_path).expanduser())
        max_size = self.parameters.get('max_size',None)
        mode = self.parameters.get('mode','mean')
        if isinstance(mode,list):
            mode = np.expand_dims(np.array(mode),axis=[0,2])
        output_column = self.parameters.get('output_column','wav2vec2')
        layer = self.parameters.get('layer','output')
        save_feature_files = self.parameters.get('save_feature_files',False)
        if save_feature_files:
            feature_output_path = GenericFile(self.global_parameters['cache_path'],self.task_hash,'features')
            feature_output_path.mkdir(parents=True)

        wav2vec_model,model_config,task = self.build_wav2vec_model(model_path,dict_path)
        wav2vec_model.cuda()

        def extract_embedding(filename):
            output_feature_filename = GenericFile(feature_output_path,'{}.feat'.format(Path(filename).stem))
            if output_feature_filename.exists() and self.cache:
                if save_feature_files:
                    return output_feature_filename
                else:
                    return output_feature_filename.load()
            else:
                x,fs = librosa.core.load(filename, sr=16000)
                if max_size:
                    x = x[:max_size]
                x = torch.tensor(np.expand_dims(x,axis=0)).cuda()
                activations = wav2vec_model.extract_features(x, None)

                if layer == 'output':
                    features = activations[-1].cpu().detach().numpy()
                elif layer == 'local_encoder':
                    features = activations[0].cpu().detach().numpy()
                elif layer == 'transformer_layers':
                    features = [activations[i].cpu().detach().numpy() for i in range(1,len(activations))]
                elif layer == 'enc_and_transformer':
                    features = [activation.cpu().detach().numpy() for activation in activations]
                elif isinstance(layer,list):
                    features = [activations[i].cpu().detach().numpy() for i in layer]
                elif isinstance(layer,dict):
                    layer_from = layer.get('from',0)
                    layer_to = layer.get('to',len(activations))
                    features = activations[layer_from:layer_to]
                    features = [f.cpu().detach().numpy() for f in features]
                
                if not isinstance(features,list):
                    features = [features]

                features = np.concatenate(features,axis=1)

                if mode == 'mean':
                    features = np.mean(features,axis=0).astype(np.float32)
                elif mode == 'sequence':
                    pass
                elif type(mode).__name__=='ndarray':
                    features = np.mean(mode*features,axis=1).astype(np.float32)

                if save_feature_files:
                    joblib.dump(features, output_feature_filename.local_filename)
                    output_feature_filename.upload_from(output_feature_filename.local_filename)
                    return output_feature_filename
                else:
                    return features

        tqdm.pandas()

        data[output_column] = data['filename'].progress_apply(extract_embedding)
        return data