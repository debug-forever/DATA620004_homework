import numpy as np

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        sigmoid_x = ActivationFunctions.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x):
        # 防止数值溢出
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """
        初始化三层神经网络
        
        参数:
            input_size: 输入特征数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出类别数量
            activation: 激活函数类型，可选'relu', 'sigmoid', 'tanh'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 权重初始化 - 使用He初始化或Xavier初始化
        if activation == 'relu':
            # He初始化适用于ReLU激活函数
            scale1 = np.sqrt(2.0 / input_size)
            scale2 = np.sqrt(2.0 / hidden_size)
        else:
            # Xavier初始化适用于sigmoid和tanh激活函数
            scale1 = np.sqrt(1.0 / input_size)
            scale2 = np.sqrt(1.0 / hidden_size)
        
        # 第一层权重和偏置
        self.W1 = scale1 * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        
        # 第二层权重和偏置
        self.W2 = scale2 * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        
        # 设置激活函数
        if activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
        else:
            raise ValueError("不支持的激活函数类型")
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据, shape (batch_size, input_size)
            
        返回:
            y_pred: 预测结果, shape (batch_size, output_size)
            cache: 缓存中间结果用于反向传播
        """
        # 第一层
        z1 = X.dot(self.W1) + self.b1
        a1 = self.activation(z1)
        
        # 第二层（输出层）
        z2 = a1.dot(self.W2) + self.b2
        y_pred = ActivationFunctions.softmax(z2)
        
        # 缓存中间结果用于反向传播
        cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'y_pred': y_pred
        }
        
        return y_pred, cache
    
    def backward(self, y_true, cache, reg_lambda=0.01):
        """
        反向传播计算梯度
        
        参数:
            y_true: 真实标签, shape (batch_size, output_size)，one-hot编码
            cache: 前向传播的中间结果
            reg_lambda: L2正则化强度
            
        返回:
            gradients: 包含所有参数梯度的字典
        """
        X = cache['X']
        a1 = cache['a1']
        z1 = cache['z1']
        y_pred = cache['y_pred']
        
        batch_size = X.shape[0]
        
        # 计算输出层误差
        dz2 = y_pred - y_true  # 交叉熵损失对softmax的导数
        
        # 计算第二层参数的梯度
        dW2 = (1/batch_size) * a1.T.dot(dz2) + reg_lambda * self.W2
        db2 = (1/batch_size) * np.sum(dz2, axis=0)
        
        # 计算反向传播到隐藏层的误差
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * self.activation_derivative(z1)
        
        # 计算第一层参数的梯度
        dW1 = (1/batch_size) * X.T.dot(dz1) + reg_lambda * self.W1
        db1 = (1/batch_size) * np.sum(dz1, axis=0)
        
        gradients = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }
        
        return gradients
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 输入数据, shape (batch_size, input_size)
            
        返回:
            预测的类别索引, shape (batch_size,)
        """
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def save_weights(self, file_path):
        """保存模型权重到文件"""
        weights = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        np.save(file_path, weights)
        
    def load_weights(self, file_path):
        """从文件加载模型权重"""
        weights = np.load(file_path, allow_pickle=True).item()
        
        # 检查模型结构是否匹配
        assert self.input_size == weights['input_size'], "输入大小不匹配"
        assert self.hidden_size == weights['hidden_size'], "隐藏层大小不匹配"
        assert self.output_size == weights['output_size'], "输出大小不匹配"
        
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2'] 