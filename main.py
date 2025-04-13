import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    """主程序入口"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='CIFAR-10图像分类神经网络')
    parser.add_argument('mode', choices=['train', 'test', 'search', 'visualize'], help='程序运行模式')
    parser.add_argument('--hidden_size', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--activation', choices=['relu', 'sigmoid', 'tanh'], default='relu', help='激活函数类型')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='学习率衰减率')
    parser.add_argument('--reg_lambda', type=float, default=0.01, help='L2正则化强度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.npy', help='模型权重路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据运行模式执行不同的功能
    if args.mode == 'train':
        # 导入训练所需的模块
        from train import main as train_main
        train_main()
    
    elif args.mode == 'test':
        # 导入测试所需的模块
        from test import main as test_main
        test_main()
    
    elif args.mode == 'search':
        # 导入超参数搜索所需的模块
        from hyperparameter_search import main as search_main
        search_main()
    
    elif args.mode == 'visualize':
        # 导入数据可视化所需的模块
        from data_utils import load_cifar10, visualize_samples
        
        print("加载CIFAR-10数据集样本进行可视化...")
        train_data, train_labels, _, _ = load_cifar10()
        
        # 随机选择一些样本进行可视化
        indices = np.random.choice(len(train_data), 25, replace=False)
        samples = train_data[indices]
        sample_labels = train_labels[indices]
        
        # 可视化样本
        visualize_samples(samples, sample_labels)

if __name__ == "__main__":
    main() 