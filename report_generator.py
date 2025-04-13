import os
import json
import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNet
from data_utils import load_cifar10, preprocess_data
from test import plot_confusion_matrix

def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training curves for loss and accuracy
    
    Parameters:
        history: Training history
        save_path: Path to save the image
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to '{save_path}'")

def visualize_network_params(model, save_dir='./report'):
    """
    Visualize network parameters
    
    Parameters:
        model: Neural network model
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Visualize first layer weights
    plt.figure(figsize=(15, 10))
    n_neurons = min(16, model.W1.shape[0])  # Show at most 16 neurons
    
    # Calculate the size of each weight vector
    weight_size = model.W1.shape[1]
    img_size = int(np.sqrt(weight_size / 3))  # Assuming input images are square
    
    for i in range(n_neurons):
        plt.subplot(4, 4, i+1)
        # Reshape weight vector to image shape
        try:
            weight_img = model.W1[i].reshape(img_size, img_size, 3)
            # Normalize weights for visualization
            weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min() + 1e-8)
            plt.imshow(weight_img)
        except ValueError:
            # If reshaping fails, show weight distribution instead
            plt.hist(model.W1[i], bins=20)
            plt.title(f'Neuron {i+1} Weight Distribution')
        plt.axis('off')
        if 'title' not in plt.gca().get_title():
            plt.title(f'Neuron {i+1}')
    
    plt.suptitle('First Layer Weight Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'first_layer_weights.png'))
    plt.close()
    
    # Visualize second layer weights
    plt.figure(figsize=(15, 10))
    n_neurons = min(16, model.W2.shape[0])
    for i in range(n_neurons):
        plt.subplot(4, 4, i+1)
        # Show weight distribution for second layer
        plt.hist(model.W2[i], bins=20)
        plt.title(f'Neuron {i+1}')
        plt.axis('off')
    
    plt.suptitle('Second Layer Weight Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'second_layer_weights.png'))
    plt.close()

def generate_experiment_report(model_path='./checkpoints/best_model.npy', 
                              history_path='training_history.json',
                              save_dir='./experiment_report'):
    """
    Generate experiment report
    
    Parameters:
        model_path: Path to model weights
        history_path: Path to training history
        save_dir: Directory to save report
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load model
    input_size = 3072  # CIFAR-10 image size
    hidden_size = 128  # Default hidden layer size
    output_size = 10   # CIFAR-10 classes
    
    model = ThreeLayerNet(input_size, hidden_size, output_size, activation='relu')
    model.load_weights(model_path)
    
    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # 1. Plot training curves
    plot_training_curves(history, save_path=os.path.join(save_dir, 'training_curves.png'))
    
    # 2. Visualize network parameters
    visualize_network_params(model, save_dir=os.path.join(save_dir, 'network_params'))
    
    # 3. Generate confusion matrix
    # Load test data
    _, _, test_data, test_labels = load_cifar10()
    test_data, test_labels = preprocess_data(test_data, test_labels)
    
    # Evaluate model on test set
    y_pred, _ = model.forward(test_data)
    predictions = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_classes, predictions, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # 4. Generate experiment report text
    report_text = f"""
# CIFAR-10 Image Classification Experiment Report

## 1. Model Introduction

This experiment uses a three-layer neural network for image classification on the CIFAR-10 dataset. The network structure is as follows:
- Input Layer: 3072 neurons (32x32x3)
- Hidden Layer: {hidden_size} neurons with ReLU activation function
- Output Layer: 10 neurons with Softmax activation function

## 2. Dataset Introduction

CIFAR-10 is an image dataset containing 10 classes, with 6000 32x32 color images per class.
- Training Set: 50000 images
- Test Set: 10000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 3. Training Process

The following hyperparameters were used during training:
- Learning Rate: 0.01
- Learning Rate Decay: 0.95
- Batch Size: 128
- Epochs: 20
- L2 Regularization Strength: 0.01

The loss and accuracy changes during training are shown in the following figure:

![Training Curves](training_curves.png)

## 4. Network Parameter Analysis

### 4.1 Weight Distribution

Layer 1 Weight Distribution:

![Layer 1 Weight Distribution](network_params/first_layer_weights.png)

Layer 2 Weight Distribution:

![Layer 2 Weight Distribution](network_params/second_layer_weights.png)

### 4.2 Neuron Visualization

Layer 1 Neuron Weights Visualization:

![Layer 1 Neurons](network_params/first_layer_weights.png)

### 4.3 Weight Correlation

Layer 1 Neuron Weight Correlation:

![Weight Correlation](network_params/first_layer_weights.png)

## 5. Experimental Results

Confusion Matrix on Test Set:

![Confusion Matrix](confusion_matrix.png)

Final Test Set Accuracy: {np.mean(predictions == true_classes):.4f}
    """
    
    # Save experiment report
    with open(os.path.join(save_dir, 'experiment_report.md'), 'w') as f:
        f.write(report_text)
    
    print(f"Experiment report generated and saved to {save_dir} directory")

if __name__ == "__main__":
    generate_experiment_report() 