from IPython import embed
from paips.core import Task
from pathlib import Path
import pandas as pd
import tqdm
import glob
import re
import numpy as np
from scipy.stats import entropy
from collections import Counter
import librosa

def list2entropy(EMOTIONS_LIST, base=None):
    value, counts = np.unique(EMOTIONS_LIST, return_counts=True)
    return entropy(counts, base=base)

def ann2prob(x, EMOTIONS_LIST):
    c = Counter(x)
    c = [c[i] for i in EMOTIONS_LIST]
    return c / np.sum(c)

FRAMES_PER_SEC = 100
AV_DICT = {True: '+', False: '-'}
RENAMING = {'neu': 'neutral', 'ang': 'anger', 'fea': 'fear', 'exc': 'excited', 'sad': 'sadness', 'fru': 'frustration',
            'sur': 'surprise', 'oth': 'other', 'hap': 'happiness', 'xxx': 'unassigned', 'dis': 'disgusted'}

EMOTIONS_LIST = ['neutral', 'anger', 'fear', 'excited', 'sadness', 'frustration', 'surprise', 'other', 'happiness', 'disgusted']


class IEMOCAPReader(Task):
    def process(self, data_path=None, min_duration=0.0, min_sad_frames_duration=0, sample=None,
                compute_speech_rate=False):
        
        data_path = str(Path(self.parameters['dataset_path']).expanduser().absolute())
        min_duration = self.parameters.get('min_duration',0.0)
        min_sad_frames_duration = self.parameters.get('min_sad_frames_duration',0.0)
        sample = self.parameters.get('sample',None)
        compute_speech_rate = self.parameters.get('compute_speech_rate',True)

        data = {}
        sentences_wav_files = glob.glob('{}/*/sentences/wav/*/*.wav'.format(data_path))
        for path_filename in tqdm.tqdm(sentences_wav_files):
            p = Path(path_filename)
            x, fs = librosa.core.load(p,sr=None)
            duration = x.size / fs
            logid = p.stem
            data[logid] = {'filename': path_filename, 'wavfile': p.name,
                           'modality': logid.split('_')[1].split('0')[0],
                           'modality_number': logid.split('_')[1].split('0')[1],
                           'modality_code': logid.split('_')[1],
                           'subject': p.stem.split('_')[0], 'session': int(p.stem.split('_')[0][4]),
                           'start': 0, 'end': duration, 'duration': duration}

        regex1 = r'\[(\d+\.?\d*) - (\d+\.?\d*)\]\t([^\s]*)\t([^\s]*)\t\[(\d+\.?\d*), (\d+\.?\d*), (\d+\.?\d*)\]\n'
        regex11 = r'\[\d+\.?\d* - \d+\.?\d*\]\t([^\s]*)\t[^\s]*\t\[\d+\.?\d*, \d+\.?\d*, \d+\.?\d*\]\n'
        regex2 = r'([a-zA-z]+);'

        dialog_emoevaluation_files = glob.glob('{}/*/dialog/EmoEvaluation/*.txt'.format(data_path))
        for path_filename in tqdm.tqdm(dialog_emoevaluation_files):
            with open(path_filename, 'r') as fp:
                emoeval_txt = fp.read()
            matchs = re.findall(regex1, emoeval_txt)
            for m in matchs:
                valence = float(m[4])
                activation = float(m[5])
                dominance = float(m[6])
                emotion_pair = '{},{}'.format(AV_DICT[activation > 3], AV_DICT[valence > 3])
                if activation == 3 and valence == 3:
                    emotion_pair = 'neutral'

                emotion = RENAMING[m[3]]
                data[m[2]].update({'emotion': emotion, 'valence_raw': valence, 'activation_raw': activation,
                                   'dominance_raw': dominance, 'emotion_pair': emotion_pair,
                                   'valence': emotion_pair[1], 'arousal': emotion_pair[0]})

            split = re.split(regex11, emoeval_txt)
            logids = split[1::2]
            txts = split[2::2]

            for logid, txt in zip(logids, txts):
                annotations = []
                for x in txt.splitlines():
                    if len(x) > 0:
                        if x[0] == 'C':
                            annotations_ = re.findall(regex2, x)
                            annotations.append(list(map(str.lower, annotations_)))

                if len(annotations) == 4:
                    data[logid].update({'annotations_selfreport': annotations[3]})
                else:
                    data[logid].update({'annotations_selfreport': 'nan'})
                data[logid].update({'annotations': annotations[:3]})
                data[logid].update({'annotations_proportion': {l: x for l, x in zip(
                    EMOTIONS_LIST, ann2prob(sum(annotations[:3], []), EMOTIONS_LIST))}})

        regex3 = r'(\d+)[\s]+(\d+)[\s]+(-?\d+)[\s]+(.*)\n'
        sentences_fa_files = glob.glob('{}/*/sentences/ForcedAlignment/*/*.wdseg'.format(data_path))
        for path_filename in tqdm.tqdm(sentences_fa_files):
            p = Path(path_filename)
            with open(path_filename, 'r') as fp:
                fa_wdseg = fp.read()
            matchs = re.findall(regex3, fa_wdseg)
            sad_times = []
            word_data = []

            sad_dur = 0
            for m in matchs:
                if m[3] != '<s>' and m[3] != '</s>' and m[3] != '<sil>':
                    sad_dur += int(m[1]) - int(m[0])
                    sad_times.append((float(m[0]) / FRAMES_PER_SEC, float(m[1]) / FRAMES_PER_SEC))
                    word_data.append([float(m[0]) / FRAMES_PER_SEC, float(m[1]) / FRAMES_PER_SEC, re.findall('[^()]+', m[3])[0]])

            if len(sad_times) == 0:
                sad_times = pd.np.nan

            data[p.stem].update({'sad_times': sad_times, 'sad_dur': sad_dur, 'words_data': word_data})

        if compute_speech_rate:
            regex3 = r'(\d+)[\s]+(\d+)[\s]+(.*)\n'
            syseg_files = glob.glob('{}/*/sentences/ForcedAlignment/*/*.syseg'.format(data_path))
            for path_filename in tqdm.tqdm(syseg_files):
                p = Path(path_filename)
                with open(path_filename, 'r') as fp:
                    fa_syseg = fp.read()
                matchs = re.findall(regex3, fa_syseg)
                dsyl = [int(m[1]) - int(m[0]) for m in matchs]
                syl_times = [(float(m[0]) / FRAMES_PER_SEC, float(m[1]) / FRAMES_PER_SEC) for m in matchs]
                if len(dsyl) != 0:
                    syl_avg_dur = np.mean(dsyl) / FRAMES_PER_SEC
                else:
                    syl_avg_dur = np.nan

                data[p.stem].update({'syl_avg_dur': syl_avg_dur, 'syl_avg_rate': 1.0 / syl_avg_dur, 'syl_times': syl_times})

        df_data = pd.DataFrame.from_dict(data, orient='index')

        df_data['annotations_entropy'] = df_data.annotations.map(lambda x: list2entropy(x[:3]))
        df_data['annotations_entropy_self_report'] = df_data.annotations.map(lambda x: list2entropy(x))
        df_data['annotations_full_agreement'] = df_data['annotations_entropy'] == 0.0

        df_data.drop(df_data[df_data.isna().any(1)].index, inplace=True)

        # Full sentences
        df_data['sentences'] = df_data.words_data.map(lambda x: ' '.join(map(lambda x: x[2], x)))

        print('Removing ' + str((df_data.duration <= min_duration).sum()) +
                    ' samples with less duration than ' + str(min_duration) + ' seconds')
        print('Removing ' + str((df_data.sad_dur <= min_sad_frames_duration).sum()) +
                    ' samples with less speech activity than ' + str(min_sad_frames_duration) + ' frames')

        df_data = df_data[(df_data.duration > min_duration)]
        df_data = df_data[(df_data.sad_dur > min_sad_frames_duration)]

        if sample is not None:
            df_data = df_data.sample(sample, random_state=12345)

        return df_data.sort_index()