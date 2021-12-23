### Repository for the paper 'Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings' by Leonardo Pepino, Pablo Riera and Luciana Ferrer

##### Requirements:

We recommend running these scripts using a virtual environment like Anaconda, which should have Tensorflow 2.4.1 and PyTorch 1.7.1 installed.

Install required python packages:
```
pip install -r requirements.txt
```

Install sox and libmediainfo in your system
```
sudo apt-get install sox
sudo apt-get install libmediainfo-dev
```

[RAVDESS](https://zenodo.org/record/1188976#.YILiD3VKiV4) and [IEMOCAP](https://sail.usc.edu/iemocap/) datasets need to be downloaded and placed at ~/Datasets with a folder structure like this:
```
├── IEMOCAP
│   ├── Documentation
│   ├── README.txt
│   ├── Session1
│   ├── Session2
│   ├── Session3
│   ├── Session4
│   └── Session5
└── RAVDESS
    └── RAVDESS
        ├── song
        └── speech
```

##### Replicating our experiments

In our paper we run many different experiments using 5 seeds for each one. If you want to replicate that procedure,
run in a terminal:

```
./run_seeds.sh <output_path>
```

If you want to run just 1 seed:

```
./run_paper_experiments.sh <seed_number> <output_path>
```

If you don't want to run all the experiments performed in the paper, comment the unwanted experiments in the run_paper_experiments.sh script. For example, our best performing model is trained using the following lines:

```sh
#w2v2PT-fusion
errors=1
while (($errors!=0)); do
paiprun configs/main/w2v2-os-exps.yaml --output_path "${OUTPUT_PATH}/w2v2PT-fusion/${SEED}" --mods "${seed_mod}&global/wav2vec2_embedding_layer=enc_and_transformer&global/normalize=global"
errors=$?; done
```

The experiments outputs will be saved at <output_path>. A cache folder will be generated at the directory from which above line is called.
Take into account that run_seeds.sh executes many experiments (all the presented in the paper), and repeats it 5 times (using different seeds for the random number generators), so it is expected that the process
takes a very long time and drive space. We ran the experiments using multiple AWS P3.2x large instances, which have a Tesla V100 GPU.

##### Analyzing the outputs

The outputs saved at <output_path> can be examined from Python using joblib. For example, running:

```python
import joblib
metrics = joblib.load('experiments/w2v2PT-fusion/0123/MainTask/DownstreamRavdess/RavdessMetrics/out')
```

will load the resulting metrics in the 'metrics' variable.

In this [notebook](notebooks/results.ipynb), more examples of how the generated outputs can be analysed are given.
Moreover, we provide the results from all our experiments in the experiments folder and the results.ipynb notebook will generate the tables of our paper.

### Using pretrained models

