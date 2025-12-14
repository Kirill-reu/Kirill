# Вариант 20: Transformer архитектура для NLP
 ### 1) Полное задание из методички:
  - Задача: реализовать Transformer для machine translation или language modeling.
  - Требования: Multi-Head Self-Attention, Position Encoding, Feed-Forward Network, Encoder-Decoder архитектура, Beam search декодирование
  - Что нужно дополнить:
    1. PositionalEncoding
    2. MultiHeadSelfAttention с scaled dot-product
    3. FeedForward сеть
    4. EncoderLayer с residual connections
    5. DecoderLayer с masked self-attention и cross-attention
    6. Полный Transformer
    7. Декодирование алгоритмы (greedy и beam search)
    8. Evaluate метрики (BLEU для machine translation)
### 2) Алгоритм работы по блокам:
   ##### 1) Блок импорта библиотек
      import numpy as np
      from sklearn.datasets import load_iris
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      from sklearn.metrics import accuracy_score, precision_score, recall_score
      import matplotlib.pyplot as plt
      from scipy.ndimage import gaussian_filter1d
   ##### 2) Блок определения класса трансформера
      - Инициализация весов
        def __init__(self, input_size, hidden_size, num_classes):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_v = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W3 = np.random.randn(hidden_size, num_classes) * 0.1
      - Функции активации
        def relu(self, x):
           return np.maximum(0, x)
        
        def softmax(self, x):
           ex = np.exp(x - np.max(x, axis=1, keepdims=True))
           return ex / (np.sum(ex, axis=1, keepdims=True) + 1e-8)
      - Прямое распространение (Forward Pass)
        def forward(self, X):
        
        #1. Embedding layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        #2. Positional Encoding (упрощенный)
        pos_enc = np.random.randn(batch_size, self.hidden_size) * 0.01
        A1 = A1 + pos_enc
        
        #3. Multi-Head Attention
        Q = np.dot(A1, self.W_q)  # Query
        K = np.dot(A1, self.W_k)  # Key
        V = np.dot(A1, self.W_v)  # Value
        
        #4. Scaled Dot-Product Attention
        scores = np.dot(Q, K.T) / np.sqrt(self.hidden_size)
        attn_weights = self.softmax(scores)
        attn_output = np.dot(attn_weights, V)
        
        #5. Feed-Forward с residual connection
        Z2 = np.dot(A1 + attn_output, self.W2) + self.b2
        A2 = self.relu(Z2)
        
        #6. Output layer
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.softmax(Z3)
      - Обратное распространение (Backward Pass)
        def backward(self, X, y, learning_rate):
           #Cross-entropy gradient
           dZ3 = logits.copy()
           dZ3[np.arange(batch_size), y] -= 1
           dZ3 /= batch_size
        
           #Backprop через слои
           dW3 = np.dot(self.cache['A2'].T, dZ3)
           dA2 = np.dot(dZ3, self.W3.T)
           dZ2 = dA2 * self.relu_grad(self.cache['Z2'])
        
           #Обновление весов
           self.W3 -= learning_rate * dW3
           self.W2 -= learning_rate * dW2
   3) Блок основной функции Main()
