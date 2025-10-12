# genXPS

## Overview

This repository contains scripts and utilities for processing XPS spectra, training classifiers, and generating predicted spectra. The main functionality is provided in the `notebooks_clean` folder:

1. **Data Processing**
   - `xps-datagen.ipynb`: Scripts for preparing experimental and synthetic XPS data for training and evaluation.

2. **Classifier Training**
   - `classifier_training.ipynb`: Trains three types of models — a conventional MLP, a CNN, and a novel Spatial Transformer Network (STN) — for multi-label functional group classification.

3. **Classifier Testing**
   - `classifier_testing.ipynb`: Evaluates trained classifiers on held-out test data.

4. **Generative Model Training**
   - `CVAE_training.ipynb`: Trains a conditional Variational Autoencoder (CVAE) to generate predicted XPS spectra from input labels.

5. **Auxiliary Scripts**
   - `explainer.ipynb`: Investigates the classifier’s prediction mechanism using gradient-based methods.  
   - `peak_annotator.ipynb`: Implements automatic peak assignment of input spectra. The workflow uses the classifier to identify present functional groups and the generative model to generate counterfactual spectra, allowing identification of which peaks correspond to which functional groups.

**Note:** Legacy scripts (not refactored) are included in `notebooks_legacy` for reference.


## Installation

This project uses [UV](https://github.com/fpgmaas/uv) to manage the Python environment and dependencies.  

### 1. Install UV
If you don't have UV installed, you can install it with pip:
```bash

pip install uv
```

### 2. Set up the virtual environment

To create a virtual environment with all required dependencies, run:

```bash
uv install
```

3. Activate the environment

```bash 
source .venv/bin/activate
```

## Data disclaimer 

Trained models are not uploaded as they exceed githubs file size limit, but are available upon reasonable request.

Input experimental polymer XPS data used for this work is excluded from the repo as it is licensed, but is available from https://surfacespectra.com/xps/


