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
      from scipy.ndimage import gaussian_filter1d>
   ##### 2) Блок определения класса трансформера
