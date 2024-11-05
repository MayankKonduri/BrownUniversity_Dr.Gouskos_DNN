# GouskosLab Project Summary (January 2024 - Present)

> **Important Note:** This is partial code, cannot share all without approval.


## Graph-Based Deep Neural Network for k-Nearest Neighbors (kNN) Visualization

This repository contains an implementation of a Graph Neural Network (GNN) using PyTorch Geometric, along with a method for visualizing k-nearest neighbor (kNN) graphs using NetworkX and Matplotlib. The project is built in Python and can be executed in Google Colab or a local environment with GPU support.

## Features
- **Graph Neural Network (GNN) Implementation**: Utilizes GCNConv layers for effective graph representation.
- **kNN Graph Visualization**: Visualize k-nearest neighbor graphs using NetworkX for intuitive understanding of data relationships.
- **Custom Data Generation**: Generate synthetic datasets or use your own dataset for flexibility in experimentation.
- **Training and Evaluation Pipeline**: Complete pipeline with the capability to visualize graph structures during the training process.

## Particle Identification with Teacher Model

This project also explores particle identification by leveraging a teacher model to enhance resolution accuracy. The teacher model provides high-quality guidance during training, allowing the student model to learn more effectively from complex datasets. This dual-model approach significantly improves the identification performance of particles in challenging scenarios, enabling more precise classifications based on graph representations.

![image](https://github.com/user-attachments/assets/1bd22a69-a493-4055-b3f0-89012fd753fb)
![download (2)](https://github.com/user-attachments/assets/d6cc8ac8-27eb-4c47-9b04-67f6c6b4fc2c)



## References

[1]  Eric Eaton. *Using multiresolution learning for transfer in image classification.* (1):1–2, 2007.

## Installation

To install the required libraries, run:

```bash
pip install torch torchvision torchaudio torch-geometric networkx matplotlib

git clone https://github.com/yourusername/gnn-knn-visualization.git

cd gnn-knn-visualization


