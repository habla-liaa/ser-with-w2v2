from paips.core import Task
import glob
from pathlib import Path
from pymediainfo import MediaInfo
import tqdm
import pandas as pd
import re

from IPython import embed

class AudioDatasetFromDirectory(Task):
    def extract_info_from_file(self,file):
        try:
            file_dict = {}
            stem_col = self.parameters.get('stem_as_column',None)
            filename_col = self.parameters.get('filename_as_column',None)
            dir_col = self.parameters.get('parents_as_columns',None)
            pos_col = self.parameters.get('positions_as_columns',None)
            audio_info_fields = self.parameters.get('audio_info_fields',['sampling_rate','duration','samples_count','channel_s'])
            filename_re = self.parameters.get('filename_re',None)
            if stem_col:
                file_dict[stem_col] = file.stem
            if filename_col:
                file_dict[filename_col] = file.name
            if dir_col:
                if isinstance(dir_col,list):
                    for i,level in enumerate(reversed(dir_col)):
                        if level is not None:
                            file_dict[level] = file.parts[-2-i]
                elif isinstance(dir_col,dict):
                    for i,level in dir_col.items():
                        file_dict[level] = file.parts[-1-int(i)]
            if pos_col:
                for i,level in enumerate(pos_col):
                    if level is not None:
                        file_dict[level] = file.parts[-1-int(i)]
            if filename_re:
                re_match = re.match(filename_re,file.name)
                if re_match is not None:
                    fields = re_match.groupdict()
                else:
                    target_field = re.search('.*<(.*)>.*',filename_re).group(1)
                    fields = {target_field: ''}
                file_dict.update(fields)
            file_dict['filename'] = str(file.absolute())
            if audio_info_fields:
                info = MediaInfo.parse(str(file.absolute()))
                audiotrack = None
                for track in info.tracks:
                    if track.track_type == 'Audio':
                        audiotrack = track
                        break

                for field in audio_info_fields:
                    file_dict[field] = getattr(audiotrack,field)
                file_dict['samples_count_from_duration'] = int(getattr(audiotrack,'sampling_rate')*getattr(audiotrack,'duration')/1000)
            file_dict['end'] = file_dict['samples_count_from_duration']
            file_dict['start'] = 0
            #files.append(file_dict)
            return file_dict
        except:
            print('An error happened when trying to read {}'.format(file))
            return None

    def process(self):
        files = []
        ext = self.parameters.get('extension','wav')
        files = list(Path(self.parameters['dataset_path']).expanduser().rglob('*.{}'.format(ext)))
        if self.parameters.get('run_parallel',False):
            from ray.util.multiprocessing.pool import Pool
            import os
            def set_niceness(niceness): # pool initializer
                os.nice(niceness)
            pool = Pool(initializer=set_niceness,initargs=(20,)) #(Run in same host it was called)
            files_metadata = pool.map(self.extract_info_from_file,files)

        else:
            files_metadata = [self.extract_info_from_file(file) for file in tqdm.tqdm(files)]
            files_metadata = [f for f in files_metadata if f is not None]

        output_df = pd.DataFrame(files_metadata)

        extra_files = self.parameters.get('extra_files',None)
        metadata_files = self.parameters.get('metadata_files',None)
        partition_lists = self.parameters.get('partition_lists',None)

        if extra_files:
            for k,v in extra_files.items():
                all_files = glob.glob('{}/{}/*.{}'.format(self.parameters['dataset_path'],v['path'],v['extension']))
                temp_df = pd.DataFrame({'{}_filename'.format(k):all_files, 'stem': [Path(f).stem for f in all_files]})
                output_df['stem'] = output_df['file_name'].apply(lambda x: Path(x).stem)
                output_df = pd.merge(output_df,temp_df,left_on='stem',right_on='stem').drop('stem',axis=1)

        if metadata_files:
            subsets = []
            for metadata_name, metadata_params in metadata_files.items():
                csv_kwargs = metadata_params.get('read_csv_kwargs',{})
                if metadata_params['path'].startswith('http:/') or metadata_params['path'].startswith('https:/'):
                    df_metadata = pd.read_csv(metadata_params['path'],**csv_kwargs)
                else:
                    df_metadata = pd.read_csv(Path(self.parameters['dataset_path'],metadata_params['path']),**csv_kwargs)
                subset_i = output_df.loc[output_df[metadata_params['merge_column_others']].isin(df_metadata[metadata_params['merge_column_self']])]
                subset_merged = pd.merge(subset_i,df_metadata,how='left',left_on=metadata_params['merge_column_others'],right_on=metadata_params['merge_column_self'])  
                subsets.append(subset_merged)
            output_df = pd.concat(subsets)
            output_df.reset_index().drop('index',axis=1)

        if partition_lists:
            for part_config in partition_lists:
                if part_config['filename'] == 'default':
                    output_df['partition'] = part_config['partition']
            for part_config in partition_lists:
                path_match = part_config.get('match_criteria','dir')
                if part_config['filename'] != 'default':
                    reference_column = part_config.get('reference_column','filename')
                    list_path = Path(part_config['filename']).expanduser()
                    if list_path.suffix == '.csv':
                        list_csv = pd.read_csv(list_path)
                        part_idxs = list_csv[part_config.get('csv_column','filename')].values
                    else:
                        with open(str(list_path),'r') as f:
                            part_idxs = f.read().splitlines()
                    if path_match == 'dir':
                        output_df = output_df.set_index(reference_column)
                        abs_paths = [str(Path(self.parameters['dataset_path'],idx).expanduser().absolute()) for idx in part_idxs]
                        output_df.loc[output_df.index.isin(abs_paths),'partition']=part_config['partition']
                        output_df = output_df.reset_index()
                    elif path_match == 'filename':
                        output_df.loc[output_df.filename.apply(lambda x: Path(x).name).isin(part_idxs),'partition']=part_config['partition']

        downsample = self.parameters.get('downsample',None)
        if downsample:
            if downsample <1:
                output_df = output_df.sample(frac=downsample)
            else:
                output_df = output_df.sample(downsample)

        return output_df