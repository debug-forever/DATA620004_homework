import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from model import ThreeLayerNet
from loss import Loss, Optimizer
from data_utils import (
    load_cifar10, preprocess_data
)

def generate_batches(data, labels, batch_size):
    """
    Generate batches of data and labels
    
    Parameters:
        data: Input data
        labels: Input labels
        batch_size: Size of each batch
        
    Yields:
        batch_data: Batch of data
        batch_labels: Batch of labels
    """
    n_samples = data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        yield batch_data, batch_labels

def create_validation_split(data, labels, val_ratio=0.1):
    """
    Create validation split from training data
    
    Parameters:
        data: Training data
        labels: Training labels
        val_ratio: Ratio of validation set size to total training set size
        
    Returns:
        train_data: Training data
        train_labels: Training labels
        val_data: Validation data
        val_labels: Validation labels
    """
    n_samples = data.shape[0]
    n_val = int(n_samples * val_ratio)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]
    
    # Split data
    val_data = data_shuffled[:n_val]
    val_labels = labels_shuffled[:n_val]
    train_data = data_shuffled[n_val:]
    train_labels = labels_shuffled[n_val:]
    
    return train_data, train_labels, val_data, val_labels

def train_model(
    model, 
    train_data, 
    train_labels, 
    val_data, 
    val_labels, 
    batch_size=128, 
    epochs=10, 
    learning_rate=0.01,
    lr_decay_rate=0.95,
    lr_decay_steps=100,
    reg_lambda=0.01,
    checkpoint_dir='./checkpoints'
):
    """
    Train neural network model
    
    Parameters:
        model: Neural network model
        train_data: Training data
        train_labels: Training labels
        val_data: Validation data
        val_labels: Validation labels
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        lr_decay_rate: Learning rate decay rate
        lr_decay_steps: Learning rate decay steps
        reg_lambda: L2 regularization strength
        checkpoint_dir: Model checkpoint directory
        
    Returns:
        history: Training history
    """
    # Create optimizer
    optimizer = Optimizer(
        learning_rate=learning_rate,
        decay_rate=lr_decay_rate,
        decay_steps=lr_decay_steps
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    # Calculate number of batches
    n_samples = train_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Record best validation accuracy
    best_val_accuracy = 0
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Start training
    for epoch in range(epochs):
        # Train one epoch
        train_loss = 0
        train_accuracy = 0
        
        for batch_data, batch_labels in generate_batches(train_data, train_labels, batch_size):
            # Forward pass
            y_pred, cache = model.forward(batch_data)
            
            # Calculate loss
            batch_loss = Loss.cross_entropy_with_l2(y_pred, batch_labels, model, reg_lambda)
            train_loss += batch_loss
            
            # Calculate accuracy
            batch_accuracy = Loss.accuracy(y_pred, batch_labels)
            train_accuracy += batch_accuracy
            
            # Backward pass
            gradients = model.backward(batch_labels, cache, reg_lambda)
            
            # Update parameters
            optimizer.update_params(model, gradients)
            
            # Record learning rate
            history['learning_rates'].append(optimizer.learning_rate)
        
        # Calculate average training loss and accuracy
        train_loss /= n_batches
        train_accuracy /= n_batches
        
        # Evaluate model on validation set
        val_y_pred, _ = model.forward(val_data)
        val_loss = Loss.cross_entropy_with_l2(val_y_pred, val_labels, model, reg_lambda)
        val_accuracy = Loss.accuracy(val_y_pred, val_labels)
        
        # Record training history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print training progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Update best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save best model
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.npy'))
            print(f"  New best model saved with validation accuracy: {best_val_accuracy:.4f}")
    
    # Save final model
    model.save_weights('final_model.npy')
    print(f"Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Save training history to JSON file
    with open('training_history.json', 'w') as f:
        # Convert NumPy arrays to lists for JSON serialization
        serializable_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_accuracy': [float(x) for x in history['train_accuracy']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_accuracy': [float(x) for x in history['val_accuracy']],
            'learning_rates': [float(x) for x in history['learning_rates']]
        }
        json.dump(serializable_history, f, indent=2)
    print("Training history saved to 'training_history.json'")
    
    return history

def plot_training_history(history):
    """Plot training history curves"""
    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot accuracy curves
    axes[0, 1].plot(history['train_accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Plot learning rate curve
    if len(history['learning_rates']) > 0:
        # Downsample learning rates to reduce noise
        lr_indices = np.linspace(0, len(history['learning_rates'])-1, min(1000, len(history['learning_rates']))).astype(int)
        lr_values = [history['learning_rates'][i] for i in lr_indices]
        
        axes[1, 0].plot(lr_indices, lr_values)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show()
    
    # Save figure
    fig.savefig('training_history.png')
    print("Training history figure saved to 'training_history.png'")

def main():
    """Main function"""
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_data, train_labels, test_data, test_labels = load_cifar10()
    
    # Preprocess data
    print("Preprocessing training and test data...")
    train_data, train_labels = preprocess_data(train_data, train_labels)
    test_data, test_labels = preprocess_data(test_data, test_labels)
    
    # Create training and validation split
    print("Splitting training and validation sets...")
    train_data, train_labels, val_data, val_labels = create_validation_split(train_data, train_labels, val_ratio=0.1)
    
    print(f"Training set size: {train_data.shape[0]} samples")
    print(f"Validation set size: {val_data.shape[0]} samples")
    print(f"Test set size: {test_data.shape[0]} samples")
    
    # Create model
    input_size = train_data.shape[1]  # 3072 for CIFAR-10
    hidden_size = 128
    output_size = 10
    
    model = ThreeLayerNet(input_size, hidden_size, output_size, activation='relu')
    
    # Train model
    print("Starting model training...")
    history = train_model(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        batch_size=128,
        epochs=20,
        learning_rate=0.01,
        lr_decay_rate=0.95,
        lr_decay_steps=100,
        reg_lambda=0.01
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Import and use the report generator
    from report_generator import generate_experiment_report
    generate_experiment_report()

if __name__ == "__main__":
    main() 