# AI-ML-DL

A collection of scripts I've been writing while learning the fundamentals of ANNs and CNNs. Each file is a standalone experiment — I numbered them in roughly the order I worked through them, so you can follow along from the simplest models to the more involved ones.

## What's in here

| # | File | What it does |
|---|------|--------------|
| 01 | `01_simple(CNN).py` | A basic CNN — my starting point for understanding convolutions and pooling |
| 02 | `02_simple(ANN-Classification).py` | A small feed-forward network for a classification task |
| 03 | `03_ANN(DL)-Regression.py` | Same idea as above but for regression — predicting continuous values |
| 04 | `04_ANN(with PCA).py` | An ANN where I reduce the input dimensions with PCA first and see how it affects training |
| 05 | `05_A-Modern-ANN.py` | A cleaner, more modern ANN setup — better optimizer, dropout, the usual improvements |
| 06 | `06_A-modern-CNN.py` | Same treatment for the CNN — batch norm, deeper layers, modern practices |
| 07 | `07_Digit-Recognizer.py` | Digit classification (MNIST-style) — putting the CNN stuff to use |
| 08 | `08_pre-trained-ResNet.py` | Fine-tuning a pre-trained ResNet instead of training from scratch |

## Why I built this

I wanted a place to actually *write* the models myself instead of just reading about them. Each script is small and focused on one idea so I can come back later and remember what I was learning at the time. If you're also getting into deep learning, feel free to poke around — some of this might be useful as a reference.

## Running the scripts

Everything is Python. You'll need the usual suspects:
pip install torch torchvision numpy pandas scikit-learn matplotlib

Then just run whichever file you're interested in:
python 01_simple(CNN).py

A few of them expect datasets (MNIST, etc.) which get downloaded automatically the first time you run them.

## Notes

This repo is a learning journal more than a polished project — expect rough edges. I'll keep adding to it as I work through more topics.
