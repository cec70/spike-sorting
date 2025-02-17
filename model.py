# Import modules
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def build_cnn(input_shape, num_classes):
    """Build CNN model."""

    # Create a sequential model
    model = Sequential([
        # First feature layer
        Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        # Second feature layer
        Conv1D(64, kernel_size=3, activation="relu"),
        Dropout(0.3),
        # Third feature layer
        Conv1D(128, kernel_size=3, activation="relu"),
        Dropout(0.3),
        # Flatten, dense layers, and output
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def train_classifier(features, labels):
    """Train a CNN and use its extracted features to train a Random Forest classifier."""

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # CNN Training
    print("Training CNN...")
    # Reshape for CNN input
    X_train_cnn = X_train[..., np.newaxis] 
    X_test_cnn = X_test[..., np.newaxis]
    
    # One-hot encode the labels for CNN
    num_classes = len(np.unique(y_train))
    y_train_cnn = to_categorical(y_train - 1, num_classes=num_classes)
    y_test_cnn = to_categorical(y_test - 1, num_classes=num_classes)
    
    # Build and train the CNN model
    cnn_model = build_cnn(X_train_cnn.shape[1:], num_classes)
    cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test_cnn))
    
    # Evaluate CNN on the test set
    cnn_score = cnn_model.evaluate(X_test_cnn, y_test_cnn, batch_size=5)
    print("\nCNN Test accuracy: %.2f%%" % (cnn_score[1] * 100))
    
    # Extract features using the CNN's penultimate layer
    print("\nExtracting features using CNN...")
    feature_extractor = Sequential(cnn_model.layers[:-1])  # Remove the output layer
    feature_extractor.add(Flatten())  # Ensure features are flattened
    feature_extractor.compile(optimizer="adam", loss="categorical_crossentropy")
    
    # Generate CNN features for training and testing sets
    cnn_features_train = feature_extractor.predict(X_train_cnn)
    cnn_features_test = feature_extractor.predict(X_test_cnn)
    
    # Ensure labels are integers for Random Forest
    rf_y_train = y_train
    rf_y_test = y_test

    # Random Forest Training
    print("\nTraining Random Forest on CNN features...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)
    rf_model.fit(cnn_features_train, rf_y_train)
    
    # Evaluate Random Forest on the test set
    y_pred_rf = rf_model.predict(cnn_features_test)
    print("\nRandom Forest Classification Report on CNN features:")
    print(classification_report(rf_y_test, y_pred_rf))

    # Return both models (CNN and RF)
    return cnn_model, rf_model

def classify_spikes(cnn_model, rf_model, features):
    """Spikes classification using hybrid CNN-Random Forest."""

    # Reshape features for CNN input
    features_cnn = features[..., np.newaxis]
    
    # Extract CNN features
    print("Extracting features using CNN...")
    feature_extractor = Sequential(cnn_model.layers[:-1])  # Remove the output layer
    feature_extractor.add(Flatten())
    feature_extractor.compile(optimizer="adam", loss="categorical_crossentropy")
    cnn_features = feature_extractor.predict(features_cnn)
    
    # Predict class labels using the Random Forest
    print("Classifying using Random Forest...")
    predictions = rf_model.predict(cnn_features)

    return predictions