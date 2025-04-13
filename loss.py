import numpy as np

class Loss:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        """
        计算交叉熵损失
        
        参数:
            y_pred: 预测概率, shape (batch_size, num_classes)
            y_true: 真实标签(one-hot编码), shape (batch_size, num_classes)
            
        返回:
            loss: 交叉熵损失值
        """
        # 防止数值不稳定
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # 计算交叉熵
        batch_size = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        
        return loss
    
    @staticmethod
    def cross_entropy_with_l2(y_pred, y_true, model, reg_lambda):
        """
        计算带L2正则化的交叉熵损失
        
        参数:
            y_pred: 预测概率, shape (batch_size, num_classes)
            y_true: 真实标签(one-hot编码), shape (batch_size, num_classes)
            model: 神经网络模型
            reg_lambda: L2正则化强度
            
        返回:
            loss: 带L2正则化的交叉熵损失值
        """
        # 交叉熵损失
        ce_loss = Loss.cross_entropy(y_pred, y_true)
        
        # L2正则化项
        l2_reg = 0.5 * reg_lambda * (np.sum(np.square(model.W1)) + np.sum(np.square(model.W2)))
        
        return ce_loss + l2_reg
    
    @staticmethod
    def accuracy(y_pred, y_true):
        """
        计算分类准确率
        
        参数:
            y_pred: 预测概率, shape (batch_size, num_classes)
            y_true: 真实标签(one-hot编码), shape (batch_size, num_classes)
            
        返回:
            accuracy: 准确率
        """
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(pred_classes == true_classes)

class Optimizer:
    def __init__(self, learning_rate=0.01, decay_rate=0.9, decay_steps=100):
        """
        初始化SGD优化器
        
        参数:
            learning_rate: 初始学习率
            decay_rate: 学习率衰减率
            decay_steps: 学习率衰减步数
        """
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.iterations = 0
    
    def update_learning_rate(self):
        """更新学习率（指数衰减）"""
        self.iterations += 1
        self.learning_rate = self.initial_learning_rate * (self.decay_rate ** (self.iterations / self.decay_steps))
        
    def update_params(self, model, gradients):
        """
        使用计算的梯度更新模型参数
        
        参数:
            model: 神经网络模型
            gradients: 参数梯度字典
        """
        self.update_learning_rate()
        
        # 更新权重和偏置
        model.W1 -= self.learning_rate * gradients['W1']
        model.b1 -= self.learning_rate * gradients['b1']
        model.W2 -= self.learning_rate * gradients['W2']
        model.b2 -= self.learning_rate * gradients['b2'] 