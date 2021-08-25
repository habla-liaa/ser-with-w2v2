### Repository for the paper 'Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings' by Leonardo Pepino, Pablo Riera and Luciana Ferrer

##### Requirements:

We recommend running these scripts using a virtual environment like Anaconda, which should have Tensorflow 2.4.1 and PyTorch 1.7.1 installed.

Install required python packages:
```
pip install -r requirements.txt
```

Install sox in your system
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

Run in a terminal:

```
./run_seeds.sh <output_path>
```

The experiments outputs will be saved at <output_path>. A cache folder will be generated at the directory from which above line is called.
Take into account that run_seeds.sh executes many experiments (all the presented in the paper), and repeats it 5 times (using different seeds for the random number generators), so it is expected that the process
takes a very long time and drive space. We ran the experiments using multiple AWS P3.2x large instances, which have a Tesla V100 GPU.
