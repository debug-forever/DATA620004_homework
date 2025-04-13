import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from model import ThreeLayerNet
from loss import Loss, Optimizer
from data_utils import (
    load_cifar10, preprocess_data, create_validation_split, generate_batches
)

def train_with_params(
    train_data, 
    train_labels, 
    val_data, 
    val_labels, 
    hidden_size,
    activation,
    learning_rate,
    lr_decay_rate,
    reg_lambda,
    batch_size=128,
    epochs=10
):
    """
    使用给定超参数训练模型
    
    参数:
        train_data: 训练数据
        train_labels: 训练标签
        val_data: 验证数据
        val_labels: 验证标签
        hidden_size: 隐藏层大小
        activation: 激活函数类型
        learning_rate: 学习率
        lr_decay_rate: 学习率衰减率
        reg_lambda: L2正则化强度
        batch_size: 批次大小
        epochs: 训练轮数
        
    返回:
        best_val_accuracy: 最佳验证准确率
        history: 训练历史记录
    """
    # 创建模型
    input_size = train_data.shape[1]
    output_size = 10  # CIFAR-10有10个类别
    model = ThreeLayerNet(input_size, hidden_size, output_size, activation=activation)
    
    # 创建优化器
    optimizer = Optimizer(
        learning_rate=learning_rate,
        decay_rate=lr_decay_rate,
        decay_steps=100
    )
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # 计算训练批次数
    n_samples = train_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 记录最佳验证准确率
    best_val_accuracy = 0
    
    # 开始训练
    for epoch in range(epochs):
        # 训练一个轮次
        train_loss = 0
        train_accuracy = 0
        
        for batch_data, batch_labels in generate_batches(train_data, train_labels, batch_size):
            # 前向传播
            y_pred, cache = model.forward(batch_data)
            
            # 计算损失
            batch_loss = Loss.cross_entropy_with_l2(y_pred, batch_labels, model, reg_lambda)
            train_loss += batch_loss
            
            # 计算准确率
            batch_accuracy = Loss.accuracy(y_pred, batch_labels)
            train_accuracy += batch_accuracy
            
            # 反向传播
            gradients = model.backward(batch_labels, cache, reg_lambda)
            
            # 更新参数
            optimizer.update_params(model, gradients)
        
        # 计算平均训练损失和准确率
        train_loss /= n_batches
        train_accuracy /= n_batches
        
        # 在验证集上评估模型
        val_y_pred, _ = model.forward(val_data)
        val_loss = Loss.cross_entropy_with_l2(val_y_pred, val_labels, model, reg_lambda)
        val_accuracy = Loss.accuracy(val_y_pred, val_labels)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # 更新最佳验证准确率
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
    
    return best_val_accuracy, history

def hyperparameter_search(train_data, train_labels, val_data, val_labels, results_dir='./hyperparameter_search_results'):
    """
    超参数搜索
    
    参数:
        train_data: 训练数据
        train_labels: 训练标签
        val_data: 验证数据
        val_labels: 验证标签
        results_dir: 结果保存目录
    """
    # 确保结果目录存在
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 设置要尝试的超参数
    # hidden_sizes = [64, 128, 256, 512]
    # activations = ['relu', 'sigmoid', 'tanh']
    # learning_rates = [0.01, 0.005, 0.001]
    # lr_decay_rates = [0.9, 0.95, 0.99]
    # reg_lambdas = [0.01, 0.001, 0.0001]

    hidden_sizes = [64, 128]
    activations = ['relu', 'sigmoid', 'tanh']
    learning_rates = [0.01, 0.005]
    lr_decay_rates = [0.9, 0.95]
    reg_lambdas = [0.01, 0.001]
    
    # 存储结果的列表
    results = []
    
    # 开始超参数搜索
    print("Starting hyperparameter search...")
    for hidden_size in hidden_sizes:
        for activation in activations:
            for learning_rate in learning_rates:
                for lr_decay_rate in lr_decay_rates:
                    for reg_lambda in reg_lambdas:
                        # 打印当前超参数组合
                        print(f"\nTrying hyperparameter combination:")
                        print(f"  Hidden layer size: {hidden_size}")
                        print(f"  Activation function: {activation}")
                        print(f"  Learning rate: {learning_rate}")
                        print(f"  Learning rate decay: {lr_decay_rate}")
                        print(f"  Regularization strength: {reg_lambda}")
                        
                        # 记录开始时间
                        start_time = time.time()
                        
                        # 训练模型
                        best_val_accuracy, history = train_with_params(
                            train_data, 
                            train_labels, 
                            val_data, 
                            val_labels, 
                            hidden_size=hidden_size,
                            activation=activation,
                            learning_rate=learning_rate,
                            lr_decay_rate=lr_decay_rate,
                            reg_lambda=reg_lambda,
                            batch_size=128,
                            epochs=5  # 减少轮数以加快搜索速度
                        )
                        
                        # 计算训练时间
                        training_time = time.time() - start_time
                        
                        # 记录结果
                        result = {
                            'hidden_size': hidden_size,
                            'activation': activation,
                            'learning_rate': learning_rate,
                            'lr_decay_rate': lr_decay_rate,
                            'reg_lambda': reg_lambda,
                            'best_val_accuracy': best_val_accuracy,
                            'training_time': training_time,
                            'history': {
                                'train_loss': history['train_loss'],
                                'train_accuracy': history['train_accuracy'],
                                'val_loss': history['val_loss'],
                                'val_accuracy': history['val_accuracy']
                            }
                        }
                        
                        results.append(result)
                        
                        print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
                        print(f"  Training time: {training_time:.2f} seconds")
    
    # 对结果按照验证准确率降序排序
    results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
    
    # 保存所有结果
    with open(os.path.join(results_dir, 'hyperparameter_search_results.json'), 'w') as f:
        # 将NumPy数组转换为列表以便JSON序列化
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            for key, value in result['history'].items():
                serializable_result['history'][key] = [float(v) for v in value]
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    # 输出最佳超参数组合
    best_result = results[0]
    print("\nBest hyperparameter combination:")
    print(f"  Hidden layer size: {best_result['hidden_size']}")
    print(f"  Activation function: {best_result['activation']}")
    print(f"  Learning rate: {best_result['learning_rate']}")
    print(f"  Learning rate decay: {best_result['lr_decay_rate']}")
    print(f"  Regularization strength: {best_result['reg_lambda']}")
    print(f"  Best validation accuracy: {best_result['best_val_accuracy']:.4f}")
    
    # 可视化超参数搜索结果
    visualize_hyperparameter_search_results(results, results_dir)
    
    return best_result

def visualize_hyperparameter_search_results(results, results_dir):
    """
    可视化超参数搜索结果
    
    参数:
        results: 超参数搜索结果列表
        results_dir: 结果保存目录
    """
    # 提取所有验证准确率和超参数
    val_accuracies = [result['best_val_accuracy'] for result in results]
    hidden_sizes = [result['hidden_size'] for result in results]
    learning_rates = [result['learning_rate'] for result in results]
    reg_lambdas = [result['reg_lambda'] for result in results]
    activations = [result['activation'] for result in results]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 隐藏层大小 vs 验证准确率
    hidden_size_data = {}
    for hs, va in zip(hidden_sizes, val_accuracies):
        if hs not in hidden_size_data:
            hidden_size_data[hs] = []
        hidden_size_data[hs].append(va)
    
    hs_values = sorted(hidden_size_data.keys())
    hs_means = [np.mean(hidden_size_data[hs]) for hs in hs_values]
    hs_stds = [np.std(hidden_size_data[hs]) for hs in hs_values]
    
    axes[0, 0].errorbar(hs_values, hs_means, yerr=hs_stds, marker='o')
    axes[0, 0].set_title('Hidden layer size vs Validation accuracy')
    axes[0, 0].set_xlabel('Hidden layer size')
    axes[0, 0].set_ylabel('Average validation accuracy')
    
    # 2. 学习率 vs 验证准确率
    lr_data = {}
    for lr, va in zip(learning_rates, val_accuracies):
        if lr not in lr_data:
            lr_data[lr] = []
        lr_data[lr].append(va)
    
    lr_values = sorted(lr_data.keys())
    lr_means = [np.mean(lr_data[lr]) for lr in lr_values]
    lr_stds = [np.std(lr_data[lr]) for lr in lr_values]
    
    axes[0, 1].errorbar(lr_values, lr_means, yerr=lr_stds, marker='o')
    axes[0, 1].set_title('Learning rate vs Validation accuracy')
    axes[0, 1].set_xlabel('Learning rate')
    axes[0, 1].set_ylabel('Average validation accuracy')
    
    # 3. 正则化强度 vs 验证准确率
    reg_data = {}
    for reg, va in zip(reg_lambdas, val_accuracies):
        if reg not in reg_data:
            reg_data[reg] = []
        reg_data[reg].append(va)
    
    reg_values = sorted(reg_data.keys())
    reg_means = [np.mean(reg_data[reg]) for reg in reg_values]
    reg_stds = [np.std(reg_data[reg]) for reg in reg_values]
    
    axes[1, 0].errorbar(reg_values, reg_means, yerr=reg_stds, marker='o')
    axes[1, 0].set_title('Regularization strength vs Validation accuracy')
    axes[1, 0].set_xlabel('Regularization strength')
    axes[1, 0].set_ylabel('Average validation accuracy')
    
    # 4. 激活函数 vs 验证准确率
    act_data = {}
    for act, va in zip(activations, val_accuracies):
        if act not in act_data:
            act_data[act] = []
        act_data[act].append(va)
    
    act_values = list(act_data.keys())
    act_means = [np.mean(act_data[act]) for act in act_values]
    act_stds = [np.std(act_data[act]) for act in act_values]
    
    axes[1, 1].bar(act_values, act_means, yerr=act_stds)
    axes[1, 1].set_title('Activation function vs Validation accuracy')
    axes[1, 1].set_xlabel('Activation function')
    axes[1, 1].set_ylabel('Average validation accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hyperparameter_search_visualization.png'))
    plt.show()
    print("Hyperparameter search results visualization saved to 'hyperparameter_search_visualization.png'")

def main():
    """主函数"""
    # 加载CIFAR-10数据集
    print("Loading CIFAR-10 dataset...")
    train_data, train_labels, test_data, test_labels = load_cifar10()
    
    # 预处理数据
    print("Preprocessing training and test data...")
    train_data, train_labels = preprocess_data(train_data, train_labels)
    test_data, test_labels = preprocess_data(test_data, test_labels)
    
    # 创建训练集和验证集分割
    print("Splitting training set and validation set...")
    train_data, train_labels, val_data, val_labels = create_validation_split(train_data, train_labels, val_ratio=0.1)
    
    print(f"Training set size: {train_data.shape[0]} samples")
    print(f"Validation set size: {val_data.shape[0]} samples")
    print(f"Test set size: {test_data.shape[0]} samples")
    
    # 执行超参数搜索
    best_params = hyperparameter_search(train_data, train_labels, val_data, val_labels)
    print("\nHyperparameter search completed!")
    
    # 使用最佳超参数创建和训练最终模型
    print("\nTraining final model with best hyperparameters...")
    
    # 创建模型
    input_size = train_data.shape[1]
    output_size = 10
    final_model = ThreeLayerNet(
        input_size, 
        best_params['hidden_size'], 
        output_size, 
        activation=best_params['activation']
    )
    
    # 创建优化器
    final_optimizer = Optimizer(
        learning_rate=best_params['learning_rate'],
        decay_rate=best_params['lr_decay_rate'],
        decay_steps=100
    )
    
    # 保存最佳超参数
    best_params_file = 'best_hyperparameters.json'
    with open(best_params_file, 'w') as f:
        # 创建一个没有history的副本，以便于保存
        params_to_save = {k: v for k, v in best_params.items() if k != 'history'}
        json.dump(params_to_save, f, indent=2)
    
    print(f"Best hyperparameters saved to '{best_params_file}'")

if __name__ == "__main__":
    main() 