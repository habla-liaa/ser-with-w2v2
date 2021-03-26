from paips.core import Task
import tqdm
import pandas as pd
import numpy as np
from deytah.batch_generator import BatchGenerator
from dienen import Model
from kahnfigh import Config
import os
import copy

import glob
from pathlib import Path
import joblib

import sklearn.metrics
import inspect

import seaborn as sns
import wandb
import matplotlib.pyplot as plt

from IPython import embed

class ClassificationMetrics(Task):
    def process(self):
        metrics = ['accuracy_score','balanced_accuracy_score','f1_score','precision_score','recall_score']
        macro_micro_w = ['f1_score','precision_score','recall_score']

        available_sklearn_metrics = inspect.getmembers(sklearn.metrics)
        available_sklearn_metrics = {cls[0]:cls[1] for cls in available_sklearn_metrics}

        results = {}
        for m in metrics:
            fn = available_sklearn_metrics[m]
            if m in macro_micro_w:
                for k in ['micro','macro','weighted']:
                    results[m+'_{}'.format(k)] = fn(self.parameters['targets'],self.parameters['predictions'],average=k)
            else:
                results[m] = fn(self.parameters['targets'],self.parameters['predictions'])

        labels = self.parameters.get('labels',None)
        if labels:
            labels_map = {i: labels[i] for i in range(len(labels))}
            targets = [labels_map[i] for i in self.parameters['targets']]
            predictions = [labels_map[i] for i in self.parameters['predictions']]
            results['cm'] = sklearn.metrics.confusion_matrix(targets,predictions,labels = labels)
            if results['cm'].shape[0] < 100:
                sns.heatmap(results['cm'],xticklabels=labels,yticklabels=labels,annot=True,fmt='d')
                wandb.log({'{}_ConfusionMatrix'.format(self.name): wandb.Image(plt)})
        else:
            results['cm'] = sklearn.metrics.confusion_matrix(self.parameters['targets'],self.parameters['predictions'])
            if results['cm'].shape[0] < 100:
                sns.heatmap(results['cm'],annot=True,fmt='d')
                wandb.log({'{}_ConfusionMatrix'.format(self.name): wandb.Image(plt)})
        return results