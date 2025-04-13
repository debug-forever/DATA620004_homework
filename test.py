import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNet
from loss import Loss
from data_utils import load_cifar10, preprocess_data, visualize_samples

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate model performance on test set
    
    Parameters:
        model: Neural network model
        test_data: Test data
        test_labels: Test labels
        
    Returns:
        accuracy: Model accuracy
        predictions: Model predictions
    """
    # Make predictions on test set
    y_pred, _ = model.forward(test_data)
    
    # Calculate accuracy
    accuracy = Loss.accuracy(y_pred, test_labels)
    
    # Get predicted classes
    predictions = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    return accuracy, predictions, true_classes

def visualize_predictions(test_data, true_classes, predictions, num_samples=25):
    """
    Visualize model predictions
    
    Parameters:
        test_data: Test data
        true_classes: True classes
        predictions: Predicted classes
        num_samples: Number of samples to visualize
    """
    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # If data is flattened, reshape it
    if len(test_data.shape) == 2:
        img_size = int(np.sqrt(test_data.shape[1] / 3))
        test_data_reshaped = test_data.reshape(-1, img_size, img_size, 3)
    else:
        test_data_reshaped = test_data
    
    # Randomly select samples
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    samples = test_data_reshaped[indices]
    sample_true_classes = true_classes[indices]
    sample_predictions = predictions[indices]
    
    # Create figure
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Display image
        ax.imshow(samples[i])
        
        # Set title with true and predicted class
        true_class = class_names[sample_true_classes[i]]
        pred_class = class_names[sample_predictions[i]]
        color = 'green' if sample_true_classes[i] == sample_predictions[i] else 'red'
        
        ax.set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.show()
    print("Predictions visualization saved to 'predictions_visualization.png'")

def plot_confusion_matrix(true_classes, predictions, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Parameters:
        true_classes: True classes
        predictions: Predicted classes
        save_path: Path to save the confusion matrix
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Calculate confusion matrix
    n_classes = len(class_names)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_class, pred_class in zip(true_classes, predictions):
        confusion_matrix[true_class, pred_class] += 1
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Add tick labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text to each cell
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white")
    
    # Add title and labels
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    
    # Add colorbar
    fig.colorbar(im)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to '{save_path}'")

def main():
    """Main function"""
    # Load model
    print("Loading model...")
    input_size = 3072  # CIFAR-10 image size
    hidden_size = 128  # Default hidden layer size
    output_size = 10   # CIFAR-10 classes
    
    model = ThreeLayerNet(input_size, hidden_size, output_size, activation='relu')
    model.load_weights('final_model.npy')
    
    # Load test data
    print("Loading CIFAR-10 test data...")
    _, _, test_data, test_labels = load_cifar10()
    
    # Preprocess test data
    print("Preprocessing test data...")
    test_data, test_labels = preprocess_data(test_data, test_labels)
    
    # Evaluate model
    print("Evaluating model on test set...")
    accuracy, predictions, true_classes = evaluate_model(model, test_data, test_labels)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(test_data, true_classes, predictions)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(true_classes, predictions)
    
    # Import and use the report generator
    from report_generator import visualize_network_params
    print("Visualizing network parameters...")
    visualize_network_params(model)

if __name__ == "__main__":
    main() 