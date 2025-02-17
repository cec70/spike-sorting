# Spike Sorting Project

## Description
This project implement spike sorting using multiple CI techniques. The goal is to classify neuronal spikes into different neuron types based on cortical recordings. The implementation includes preprocessing, feature extraction, and classification steps.

## Requirements
Python 3.9+ is required to run this project. The following Python libraries are also required:
- NumPy (v1.26.2)
- SciPy (v1.11.4)
- scikit-learn (v1.5.2)
- TensorFlow (v2.15.0)
- PyWavelets (v1.7.0)

Use the following command to install all dependencies:
'''bash
pip install -r requirements.txt

## How to run
The main file to execute is 'main.py'. Ensure all required dependencies are installed and the dataset files ('D1.mat' to 'D6.mat') are located in the 'Datasets/' directory.

Run the project using the following command:
'''bash
python main.py

## Output
The detected spike indices and their predicted classes are saved to '.mat' files in the 'Results/' directory.
