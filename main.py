# Import modules
import numpy as np
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score

from functions import load_mat_file, save_results
from functions import bandpass_filter, wavelet_denoising, detect_spikes, baseline_correction, extract_features
from model import train_classifier, classify_spikes

def process_dataset(dataset, file_path, output_file, train=False, model=None):
    """Main pipeline."""

    # Load .mat file
    d, Index, Class = load_mat_file(file_path)

    if train:
        # Extract features
        spike_features = extract_features(d, Index)

        # Scale features
        scaler = MinMaxScaler()
        spike_features = scaler.fit_transform(spike_features)

        # Train hybrid CNN-Random Forest classifier on D1
        cnn_model, rf_model = train_classifier(spike_features, Class)
        return cnn_model, rf_model  
    else:
        # Filter data with bandpass filter
        bpfiltered_data = bandpass_filter(d)

        # Apply median filter with kernel size depending on dataset noise
        if dataset in ["D4", "D5", "D6"]:
            filtered_data = medfilt(bpfiltered_data, kernel_size=7)
        else:
            filtered_data = medfilt(bpfiltered_data, kernel_size=3)

        # Apply wavelet denoising
        denoised_data = wavelet_denoising(filtered_data)

        # Set adaptive multipliers
        if dataset == "D2":
            multiplier = 2.23
        elif dataset == "D3":
            multiplier = 2.04
        elif dataset == "D4":
            multiplier = 2.12
        elif dataset == "D5":
            multiplier = 2.48
        elif dataset == "D6":
            multiplier = 2.37

        # Initialise threshold based on mean and standard deviation
        std = np.std(denoised_data) # Find standard deviation
        mean = np.mean(denoised_data) # Find mean
        threshold = mean + multiplier * std
        print(f"Threshold: {threshold}")

        # Detect spikes
        spike_indices = detect_spikes(denoised_data, threshold)
        print(f"Number of detected spikes: {len(spike_indices)}")

        # Apply baseline correction with window size depending on dataset noise
        if dataset in ["D5", "D6"]:
            corrected_features = baseline_correction(denoised_data, window_size=120)
        else:
            corrected_features = baseline_correction(denoised_data, window_size=100)

        # Standardise the corrected data
        scaler1 = StandardScaler()
        standardised_data = scaler1.fit_transform(corrected_features.reshape(-1, 1)).flatten()

        # Extract features
        spike_features = extract_features(standardised_data, spike_indices)

        # Scale features
        scaler2 = MinMaxScaler()
        spike_features = scaler2.fit_transform(spike_features)

        # Classify spikes using trained model
        spike_classes = classify_spikes(model[0], model[1], spike_features)

        # Display silhouette score
        silhouette_avg = silhouette_score(spike_features, spike_classes)
        print(f"Silhouette Score: {silhouette_avg}") 

        # Save results in .mat file       
        save_results(output_file, spike_indices, spike_classes)

# Execution of training dataset D1
d1_file = "Datasets/D1.mat"
cnn_model, rf_model = process_dataset("D1", d1_file, None, train=True) # Train on D1

# Execution for datasets D2 to D6
datasets = ["D2", "D3", "D4", "D5", "D6"]
for dataset in datasets:
    print(f"\nProcessing dataset {dataset}.mat...")
    process_dataset(dataset, f"Datasets/{dataset}.mat", f"Results/{dataset}.mat", train=False, model=(cnn_model, rf_model))

print("\nTraining and classification complete.")

