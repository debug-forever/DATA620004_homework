import numpy as np
import pickle
import os
import urllib.request
import tarfile
import matplotlib.pyplot as plt

def download_cifar10():
    """
    Download CIFAR-10 dataset if not already downloaded
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        
        print("Extracting files...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        print("Download and extraction complete!")
    else:
        print("CIFAR-10 dataset already downloaded.")

def load_cifar10_batch(batch_file):
    """
    Load a single batch of CIFAR-10 data
    
    Parameters:
        batch_file: Path to the batch file
        
    Returns:
        data: Image data
        labels: Image labels
    """
    with open(batch_file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        
    data = batch[b'data']
    labels = batch[b'labels']
    
    return data, labels

def load_cifar10():
    """
    Load CIFAR-10 dataset
    
    Returns:
        train_data: Training data
        train_labels: Training labels
        test_data: Test data
        test_labels: Test labels
    """
    # Check if data directory exists and contains the dataset
    data_dir = 'cifar-10-batches-py'
    if not os.path.exists(data_dir):
        print("Downloading CIFAR-10 dataset...")
        download_cifar10()
    else:
        # Check if all batch files exist
        batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        batch_files.append('test_batch')
        missing_files = [f for f in batch_files if not os.path.exists(os.path.join(data_dir, f))]
        
        if missing_files:
            print(f"Missing batch files: {missing_files}")
            print("Downloading CIFAR-10 dataset...")
            download_cifar10()
        else:
            print("CIFAR-10 dataset already exists, skipping download.")
    
    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_data, batch_labels = load_cifar10_batch(batch_file)
        train_data.append(batch_data)
        train_labels.extend(batch_labels)
    
    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)
    
    # Load test data
    test_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)
    
    return train_data, train_labels, test_data, test_labels

def preprocess_data(data, labels):
    """
    Preprocess CIFAR-10 data
    
    Parameters:
        data: Image data
        labels: Image labels
        
    Returns:
        processed_data: Normalized data
        processed_labels: One-hot encoded labels
    """
    # Normalize data
    processed_data = data.astype(np.float32) / 255.0
    
    # One-hot encode labels
    n_samples = len(labels)
    n_classes = 10
    processed_labels = np.zeros((n_samples, n_classes))
    processed_labels[np.arange(n_samples), labels] = 1
    
    return processed_data, processed_labels

def visualize_samples(data, labels, num_samples=25):
    """
    Visualize CIFAR-10 samples
    
    Parameters:
        data: Image data
        labels: Image labels
        num_samples: Number of samples to visualize
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # If data is flattened, reshape it
    if len(data.shape) == 2:
        img_size = int(np.sqrt(data.shape[1] / 3))
        data_reshaped = data.reshape(-1, img_size, img_size, 3)
    else:
        data_reshaped = data
    
    # Randomly select samples
    indices = np.random.choice(len(data), num_samples, replace=False)
    samples = data_reshaped[indices]
    sample_labels = labels[indices]
    
    # Create figure
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Display image
        ax.imshow(samples[i])
        
        # Set title with class name
        class_name = class_names[sample_labels[i]]
        ax.set_title(f"Class: {class_name}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png')
    plt.show()
    print("Sample visualization saved to 'cifar10_samples.png'")

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