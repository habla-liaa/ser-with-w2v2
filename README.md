### Repository for the paper 'Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings' by Leonardo Pepino, Pablo Riera and Luciana Ferrer

Requirements:

We recommend running these scripts using a virtual environment like Anaconda, which should have Tensorflow 2.4.1 and PyTorch 1.7.1 installed.

Install required python packages:
```
pip install -r requirements.txt
```

Install sox in your system
```
sudo apt-get install sox
```

Datasets should be placed at ~/Datasets with a structure like this:
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

Then, run the experiments by running in a terminal:

```
./run_seeds.sh <output_path>
```

The experiments outputs will be saved at <output_path>. A cache folder will be generated at the directory from which above line is called.
Take into account that run_seeds.sh executes many experiments (all the presented in the paper), and repeats it 5 times, so it is expected that the process
takes a very long time and drive space. We ran the experiments using multiple AWS P3.2x large instances, which have a Tesla V100 GPU.
