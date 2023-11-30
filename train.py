import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import model  # Import the model defined in model.py
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def train(model, fname, out_fname):
    """ Train and save the provided neural network model on Planesnet dataset

    Args:
        model (tf.keras.models.Sequential): Pre-existing Keras model
        fname (str): Path to PlanesNet JSON dataset
        out_fname (str): Path to output Keras model file (.h5)
    """

    # Load planesnet data
    f = open(fname)
    planesnet = json.load(f)
    f.close()

    # Preprocess image data and labels for input
    X = np.array(planesnet['data']) / 255.
    X = X.reshape([-1, 3, 20, 20]).transpose([0, 2, 3, 1])
    Y = np.array(planesnet['labels'])

    # Convert labels to one-hot encoding
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    Y_categorical = tf.keras.utils.to_categorical(Y_encoded, num_classes=2)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_categorical, test_size=0.2, random_state=42)

    # Train the provided model
    model.fit(X_train, Y_train, epochs=50, shuffle=True, validation_data=(X_val, Y_val), batch_size=128)

    evaluate_model(model, X_val, Y_val)

    # Save trained model
    model.save(out_fname)

def evaluate_model(model, X_val, Y_val):
    """ Evaluate the provided neural network model on validation data and display misclassified images

    Args:
        model (tf.keras.models.Sequential): Trained Keras model
        X_val (numpy.ndarray): Validation image data
        Y_val (numpy.ndarray): True labels for validation data
    """

    # Make predictions on validation data
    Y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(Y_pred, axis=1)
    y_true_labels = np.argmax(Y_val, axis=1)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Display misclassified images
    misclassified_indices = np.where(y_pred_labels != y_true_labels)[0]

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(misclassified_indices[:25]):  # Display the first 25 misclassified images
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_val[idx])
        plt.title(f'True: {y_true_labels[idx]}, Predicted: {y_pred_labels[idx]}')
        plt.axis('off')

    plt.show()

# Example usage
if __name__ == "__main__":
    # Provide the pre-existing model to the train function
    train(model, "planesnet.json", "output_model.h5")