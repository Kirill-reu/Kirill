# Реализация биноминальной кучи с помощью модуля heapq
import heapq 
someNumbers = [8, 3, 5, 1, 6, 2, 4, 7]  # создание списка с исходными данными для кучи 
heapq.heapify(someNumbers) # превращение списка в кучу 
heapq.heappush(someNumbers, 0) # добавление нового элемента в кучу 
someValue = heapq.heappop(someNumbers) # извлечение минимального элемента из кучи 
print(someNumbers) # вывод программы

# Реализация биноминальной кучи в виде собственного класса
class BinaryHeap:
    def __init__(self): 
        self.heap = [3]
    def insert(self, key): 
        self.heap.append(key) 
        self._heapify_up(len(self.heap) - 1) 
    def delete_min(self): 
        self.heap.pop() 
        self._heapify_down(0) 
    def get_min(self): 
        if self.is_empty(): 
            return None 
        return self.heap 
    def is_empty(self): 
        return len(self.heap) == 0 

# Реализация программы для печати ряда Фибоначчи до заданного количества элементов
def fibonacci_for_loop(n): 
    a, b = 0, 1 
    for _ in range(n): 
        print(a, end=' ') 
        a, b = b, a + b 
fibonacci_for_loop(10) 

# Алгоритм для рекурсивного вычисления n-го числа ряда Фибоначчи
def fibonacci(n): 
    if n in (1, 2): 
        return 1 
    return fibonacci(n - 1) + fibonacci(n - 2) 
print(fibonacci(10)) 

# Реализация с использованием словаря (dict) как хэш-таблицы
class HashTable: 
    def __init__(self, size): 
        self.size = size 
        self.table = [None]*size

    def _hash(self, key): 
        return ord(key[0]) % self.size
     
class HashTable: 
    def __init__(self, size): 
        self.size = size 
        self.table = [None]*size 

    def _hash(self, key): 
        return ord(key[0]) % self.size 
    
    def set(self, key, value): 
        hash_index = self._hash(key) 
        self.table[hash_index] = (key, value) 

    def get(self, key): 
        hash_index = self._hash(key) 
        if self.table[hash_index] is not None: 
            return self.table[hash_index][1] 
        
        raise KeyError(f'Key {key} not found')
     
    def remove(self, key): 
        hash_index = self._hash(key) 
        if self.table[hash_index] is not None: 
            self.table[hash_index] = None 
        else: 
            raise KeyError(f'Key {key} not found')
# Create a hash table of size 10 
hash_table = HashTable(10) 
# Add some key-value pairs 
hash_table.set('Alice', 'January') 
hash_table.set('Bob', 'May') 
# Retrieve a value 
print(hash_table.get('Alice'))  # Outputs: 'January' 
# Remove a key-value pair 
hash_table.remove('Bob') 
# This will raise a KeyError, as 'Bob' was removed 
print(hash_table.get('Bob')) 

# Реализации хэш-таблицы с собственной реализацией
class HashTable:
    def init(self, size=10):
        """
        Инициализирует хеш-таблицу с заданным размером.
        Внутренний массив будет хранить списки (цепочки) для разрешения коллизий.
        """
        self.array_size = size
        # Создаем массив, где каждая ячейка изначально пуста (None)
        self.array = [None] * self.array_size

    def hash_function(self, key):
        """
        Генерирует хеш-код для ключа.
        Используем встроенную функцию hash() и оператор взятия остатка,
        чтобы индекс не вышел за пределы размера массива.
        """
        return hash(key) % self.array_size

    def put(self, key, value):
        """
        Вставляет пару ключ-значение в таблицу.
        Если ключ уже существует, его значение обновляется.
        """
        index = self.hash_function(key)
        
        # Если ячейка (bucket) по этому индексу еще не создана, создаем пустой список
        if self.array[index] is None:
            self.array[index] = []
        
        bucket = self.array[index]
        
        # Проверяем, существует ли уже такой ключ в цепочке
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                # Если ключ найден, обновляем значение и выходим
                bucket[i] = (key, value)
                return
        
        # Если ключ не найден, добавляем новую пару (ключ, значение) в цепочку
        bucket.append((key, value))

    def get(self, key):
        """
        Находит и возвращает значение по ключу.
        Если ключ не найден, вызывает исключение KeyError.
        """
        index = self.hash_function(key)
        bucket = self.array[index]
        
        # Если ячейка пуста или ключа нет в цепочке, его нет в таблице
        if bucket is None:
            raise KeyError(f"Ключ '{key}' не найден.")
            
        # Ищем ключ в цепочке
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        
        # Если прошли всю цепочку и не нашли, вызываем исключение
        raise KeyError(f"Ключ '{key}' не найден.")

    def remove(self, key):
        """
        Удаляет пару ключ-значение по ключу.
        Если ключ не найден, вызывает исключение KeyError.
        """
        index = self.hash_function(key)
        bucket = self.array[index]

        if bucket is None:
            raise KeyError(f"Ключ '{key}' не найден.")

        # Ищем ключ в цепочке, чтобы удалить его
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                # Удаляем элемент из списка по его индексу
                del bucket[i]
                return
        
        # Если ключ не был найден в цепочке
        raise KeyError(f"Ключ '{key}' не найден.")

# --- Пример использования ---

# Создаем экземпляр хеш-таблицы
hash_table = HashTable(10)

# Добавляем элементы
hash_table.put("Jane Doe", "ID1234")
hash_table.put("John Smith", "ID4567")
# Пример коллизии: "Jane Doe" и "Doe Jane" могут дать один и тот же хеш-индекс
hash_table.put("Doe Jane", "ID9876") 

# Получаем значение по ключу
print(f'ID для "Jane Doe": {hash_table.get("Jane Doe")}')
# Вывод: ID для "Jane Doe": ID1234

# Удаляем элемент
hash_table.remove("John Smith")
print('"John Smith" был удален.')

# Пытаемся получить удаленный элемент
try:
    print(hash_table.get("John Smith"))
except KeyError as e:
    print(e)
    # Вывод: Ключ 'John Smith' не найден.
