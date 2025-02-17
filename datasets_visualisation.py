import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load dataset function
def load_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    d = data.get('d', None).flatten()
    return d

# Plot the 'd' vector
def plot_d_vector(file_path, dataset_name):
    d = load_mat_file(file_path)
    
    # Time vector (since the sample rate is 25 kHz)
    fs = 25000  # sample rate
    time = np.arange(0, len(d)) / fs
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, d)
    plt.title(f"Signal from {dataset_name} (d vector)")
    plt.xlabel('Time [seconds]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Main execution: Plot for D2 to D6
datasets = ['D2', 'D3', 'D4', 'D5', 'D6']
for dataset in datasets:
    file_path = f'Coursework C/Datasets/{dataset}.mat'  # Adjust the path as needed
    plot_d_vector(file_path, dataset)
