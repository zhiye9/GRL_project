# Graph Representation Learning Project
DTU Compute 2022 Graph Representation Learning Summer School
This repository contains the codes and models for the the course project of [02901 Advanced Topics in Machine Learning: Graph Representation Learning](http://www2.compute.dtu.dk/courses/02901/) (2022). The project report can be found [here](GRL_report.pdf)

### Overview

Using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAN) to classify Austism subjects from healthy control. Five models including a baseline were tested on resting-state functional MRI data from the Autism Brain Imaging Data Exchange (ABIDE) and was preprocessed by the Preprocessed Connectomes Project (PCP) (http://preprocessed-connectomes-project.org/abide/).

###Data
Data downloading and proessing were implemented on the platform of [Nilean](https://nilearn.github.io/) in Python. 

###Dataset and Models
Dataset and Models refers to the lecture material and implemented with PyTorh Geometric in Python.

### Code Structure
- df_mri_preprocess.py: Downloading and processing MRI data.
- pyg_dataset.py: Create PyG Dataset.
- GNN.py: defines the GNN architecture.
- train.py: creates the networks, harnesses and reports the progress of training.
- Install_PSM.py: Install propensity score matching method.
