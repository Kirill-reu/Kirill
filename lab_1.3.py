# Пример структуры для реализации бинарной кучи на Python

# Минимальная куча через список
class MinHeap:
    def __init__(self):
        self.heap = []

    
    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        if idx > 0 and self.heap[idx] < self.heap[parent]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            self._sift_up(parent)
    
    def push(self, val):
        self.heap.append(val)  # Динамическое расширение списка
        self._sift_up(len(self.heap) - 1)

#####################################################################################################

# Пример структуры для реализации биноминальной кучи на Python

class BinomialNode:
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.child = None    # Простые ссылки
        self.sibling = None  # Сборщик мусора управляет памятью

class BinomialHeap:
    def __init__(self):
        self.trees = []  # Список деревьев

#####################################################################################################

# Пример структуры для реализации «кучи Фибоначчи» на Python

class FibonacciNode:
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.marked = False
        self.parent = self.child = None
        self.left = self.right = self  # Циклический двусвязный список

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.count = 0
    # Сборщик мусора сам удалит изолированные узлы

#####################################################################################################

# Пример структуры для реализации «Хэш-таблицы» на Python

class HashMap:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.table = [None] * capacity  # Слоты хранят пары или None
        self.load_factor = 0.75
    
    def _hash(self, key):
        return hash(key) % self.capacity  # Встроенная хеш-функция
    
    def __setitem__(self, key, value):
        # Открытая адресация или цепочки
        idx = self._hash(key)
        if self.table[idx] is None:
            self.table[idx] = []  # Цепочка как список
        self.table[idx].append((key, value))
