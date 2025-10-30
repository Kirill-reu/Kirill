# Блочная (корзинная) сортировка

def bucket_sort(arr):
    if len(arr) == 0:
        return arr
    # Определяем диапазон значений
    min_val, max_val = min(arr), max(arr)
    # Создаем корзины
    bucket_count = len(arr)
    buckets = [[] for _ in range(bucket_count)]
    # Распределяем элементы по корзинам
    for num in arr:
        # Вычисляем индекс корзины
        index = int((num - min_val) * (bucket_count - 1) / (max_val - min_val))
        buckets[index].append(num)
    # Сортируем каждую корзину
    for i in range(bucket_count):
        buckets[i].sort()
    # Объединяем корзины
    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result

# Пример использования
if __name__ == "__main__":
    arr = [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]
    print("Исходный массив:", arr)
    sorted_arr = bucket_sort(arr)
    print("Отсортированный массив:", sorted_arr)

# Вывод:
Исходный массив: [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]
Отсортированный массив: [0.32, 0.33, 0.37, 0.42, 0.47, 0.51, 0.52]

###################################################################################################################

# Блинная сортировка

def pancake_sort(arr):
    def flip(subarray, k):
        """Переворачивает первые k элементов подмассива"""
        i = 0
        while i < k // 2:
            subarray[i], subarray[k - i - 1] = subarray[k - i - 1], subarray[i]
            i += 1
    n = len(arr)
    for curr_size in range(n, 1, -1):
        # Находим индекс максимального элемента
        max_index = 0
        for i in range(1, curr_size):
            if arr[i] > arr[max_index]:
                max_index = i
        
        if max_index != curr_size - 1:
            # Переворачиваем до максимального элемента
            if max_index != 0:
                flip(arr, max_index + 1)
            # Переворачиваем весь сегмент
            flip(arr, curr_size)
    return arr

# Пример использования
if __name__ == "__main__":
    arr = [23, 10, 20, 11, 12, 6, 7]
    print("Исходный массив:", arr)
    sorted_arr = pancake_sort(arr.copy())
    print("Отсортированный массив:", sorted_arr)

# Вывод:
Исходный массив: [23, 10, 20, 11, 12, 6, 7]
Отсортированный массив: [6, 7, 10, 11, 12, 20, 23]

#####################################################################################################################

# Cортировка бусинами (гравитационная)

def bead_sort(arr):
    if not arr:
        return []
    # Находим максимальное значение
    max_val = max(arr)
    # Создаем "абак" - матрицу бусин
    bead_grid = []
    # Инициализируем сетку бусин
    for num in arr:
        bead_grid.append([1] * num + [0] * (max_val - num))
    # "Падение" бусин под действием гравитации
    for i in range(max_val):
        # Подсчитываем бусины в каждом столбце
        beads_in_column = []
        for j in range(len(arr)):
            if i < len(bead_grid[j]):
                beads_in_column.append(bead_grid[j][i])
            else:
                beads_in_column.append(0)
        # Сортируем столбец (бусины падают вниз)
        beads_in_column.sort(reverse=True)
        # Обновляем сетку
        for j in range(len(arr)):
            if i < len(bead_grid[j]):
                bead_grid[j][i] = beads_in_column[j]
    # Считываем результат
    result = []
    for row in bead_grid:
        result.append(sum(row))
    return result

# Пример использования
if __name__ == "__main__":
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    print("Исходный массив:", arr)
    sorted_arr = bead_sort(arr)
    print("Отсортированный массив:", sorted_arr)

# Вывод:
Исходный массив: [3, 1, 4, 1, 5, 9, 2, 6]
Отсортированный массив: [9, 6, 5, 4, 3, 2, 1, 1]

######################################################################################################################

# Поиск скачками

import math

def jump_search(arr, target):
    n = len(arr)
    # Если массив пустой
    if n == 0:
        return -1
    # Определяем размер прыжка
    step = int(math.sqrt(n))
    # Находим блок, где может находиться элемент
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    # Линейный поиск в найденном блоке
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    # Проверяем, найден ли элемент
    if arr[prev] == target:
        return prev
    else:
        return -1

# Пример использования
if __name__ == "__main__":
    arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    target = 55
    print("Массив:", arr)
    print(f"Поиск элемента {target}")
    result = jump_search(arr, target)
    if result != -1:
        print(f"Элемент найден на позиции {result}")
    else:
        print("Элемент не найден")

# Вывод
Массив: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
Поиск элемента 55
Элемент найден на позиции 10

#########################################################################################################################

# Экспоненциальный поиск

def exponential_search(arr, target):
    n = len(arr)
    # Если массив пустой
    if n == 0:
        return -1
    # Если элемент в начале
    if arr[0] == target:
        return 0
    # Экспоненциальное увеличение границы
    bound = 1
    while bound < n and arr[bound] <= target:
        bound *= 2
    # Бинарный поиск в найденном диапазоне
    left = bound // 2
    right = min(bound, n - 1)
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Пример использования
if __name__ == "__main__":
    arr = [2, 3, 4, 10, 40, 45, 50, 60, 70, 80, 90, 100]
    target = 10
    print("Массив:", arr)
    print(f"Поиск элемента {target}")
    result = exponential_search(arr, target)
    if result != -1:
        print(f"Элемент найден на позиции {result}")
    else:
        print("Элемент не найден")

# Вывод
Массив: [2, 3, 4, 10, 40, 45, 50, 60, 70, 80, 90, 100]
Поиск элемента 10
Элемент найден на позиции 3

#######################################################################################################################

# Тернарный поиск

def ternary_search(arr, target):
    def ternary_search_recursive(left, right):
        if right >= left:
            # Вычисляем две точки деления
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            # Проверяем, не нашли ли элемент в точках деления
            if arr[mid1] == target:
                return mid1
            if arr[mid2] == target:
                return mid2
            # Определяем, в какой трети продолжать поиск
            if target < arr[mid1]:
                return ternary_search_recursive(left, mid1 - 1)
            elif target > arr[mid2]:
                return ternary_search_recursive(mid2 + 1, right)
            else:
                return ternary_search_recursive(mid1 + 1, mid2 - 1)
        return -1
    return ternary_search_recursive(0, len(arr) - 1)

# Пример использования
if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = 6
    print("Массив:", arr)
    print(f"Поиск элемента {target}")
    result = ternary_search(arr, target)
    if result != -1:
        print(f"Элемент найден на позиции {result}")
    else:
        print("Элемент не найден")

# Вывод
Массив: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Поиск элемента 6
Элемент найден на позиции 5