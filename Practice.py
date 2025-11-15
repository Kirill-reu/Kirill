#  Табу-поиск для задачи планирования 

import random
import copy
from collections import deque

# --- Входные данные ---
NUM_EMPLOYEES = 5
NUM_DAYS = 7
DAYS_TO_WORK = 5
MAX_CONSECUTIVE_OFF = 2

# --- Параметры Табу-поиска ---
MAX_ITERATIONS = 200  # Критерий остановки
TABU_TENURE = 7     # "Срок" запрета, сколько итераций ход остается в табу-листе

def create_initial_solution():
    """ 
    Создает начальное решение.
    Для каждого работника случайно распределяет 5 рабочих дней (1) и 2 выходных (0).
    Это гарантирует, что "Требование 1" (работать 5 дней) всегда выполнено.
    """
    schedule = []
    for _ in range(NUM_EMPLOYEES):
        days = [1] * DAYS_TO_WORK + [0] * (NUM_DAYS - DAYS_TO_WORK)
        random.shuffle(days)
        schedule.append(days)
    return schedule

def calculate_penalty(schedule):
    """
    Вычисляет "оценку качества" (штраф).
    Штраф начисляется за нарушение ограничения "не более 2 выходных подряд".
    """
    total_penalty = 0
    for emp_schedule in schedule:
        consecutive_off = 0
        for day in emp_schedule:
            if day == 0:
                consecutive_off += 1
            else:
                # Если счетчик превысил норму, начисляем штраф
                if consecutive_off > MAX_CONSECUTIVE_OFF:
                    # Штрафуем за каждый "лишний" день
                    total_penalty += (consecutive_off - MAX_CONSECUTIVE_OFF)
                consecutive_off = 0 # Сбрасываем счетчик
        
        # Проверка в конце недели, если выходные были в конце
        if consecutive_off > MAX_CONSECUTIVE_OFF:
            total_penalty += (consecutive_off - MAX_CONSECUTIVE_OFF)
            
    return total_penalty

def get_neighbors(schedule):
    """
    Генерирует "соседей" для текущего расписания.
    Сосед = расписание, где у одного сотрудника поменяли местами 
    один рабочий день (1) и один выходной (0).
    
    Возвращает список (сосед, ход), где ход = (emp_idx, work_day_idx, off_day_idx)
    """
    neighbors = []
    for emp_idx in range(NUM_EMPLOYEES):
        # Находим индексы рабочих дней и выходных для сотрудника
        work_days = [i for i, day in enumerate(schedule[emp_idx]) if day == 1]
        off_days = [i for i, day in enumerate(schedule[emp_idx]) if day == 0]
        
        # Создаем всех возможных "соседей" путем одного обмена
        for work_day_idx in work_days:
            for off_day_idx in off_days:
                # Копируем текущее расписание
                neighbor_schedule = copy.deepcopy(schedule)
                
                # Делаем "ход" (обмен)
                neighbor_schedule[emp_idx][work_day_idx] = 0
                neighbor_schedule[emp_idx][off_day_idx] = 1
                
                # "Ход" - это то, что мы добавим в табу-лист
                move = (emp_idx, work_day_idx, off_day_idx)
                neighbors.append((neighbor_schedule, move))
                
    return neighbors

def print_schedule(schedule, title="Расписание"):
    """ Удобная функция для вывода расписания """
    print(f"\n--- {title} ---")
    print("       ПН ВТ СР ЧТ ПТ СБ ВС")
    print("       -------------------")
    for i, emp_schedule in enumerate(schedule):
        # 1 = Р (Работа), 0 = В (Выходной)
        schedule_str = "  ".join(["Р" if day == 1 else "В" for day in emp_schedule])
        print(f"Сотр {i+1}: {schedule_str}")
    print("-------------------------")

# --- Основной алгоритм Табу-поиска ---

# 1. Инициализация
tabu_list = deque(maxlen=TABU_TENURE) # Используем deque для авто-удаления старых ходов
current_solution = create_initial_solution()
current_penalty = calculate_penalty(current_solution)

# Сохраняем самое лучшее найденное решение
best_solution = current_solution
best_penalty = current_penalty

print("Запускаем Табу-поиск...")
print_schedule(current_solution, "Начальное случайное расписание")
print(f"Начальный штраф: {current_penalty}")

# 2. Основной цикл
for i in range(MAX_ITERATIONS):
    if best_penalty == 0:
        print(f"\nИдеальное решение найдено на итерации {i}!")
        break
        
    # 3. Поиск лучшего соседа
    best_neighbor = None
    best_neighbor_move = None
    best_neighbor_penalty = float('inf')

    for neighbor, move in get_neighbors(current_solution):
        
        # Проверяем, в Табу-листе ли *обратный* ход? 
        # (Более надежно - проверять сам ход)
        # Если мы поменяли (i, j), то (i, j) становится табу
        
        neighbor_penalty = calculate_penalty(neighbor)
        
        if move in tabu_list:
            # 4. Аспирационный критерий
            # Разрешаем табу, если оно ведет к рекордному решению
            if neighbor_penalty < best_penalty:
                best_neighbor = neighbor
                best_neighbor_move = move
                best_neighbor_penalty = neighbor_penalty
            else:
                continue # Ход табуирован и не аспирационный, пропускаем
        
        # Ход не в табу
        elif neighbor_penalty < best_neighbor_penalty:
            best_neighbor = neighbor
            best_neighbor_move = move
            best_neighbor_penalty = neighbor_penalty

    # 5. Совершаем лучший ход (даже если он "плохой")
    if best_neighbor is None:
        # Это может случиться, если все ходы табуированы
        print("Не найдено доступных ходов. Остановка.")
        break
        
    current_solution = best_neighbor
    current_penalty = best_neighbor_penalty
    
    # 6. Обновляем Табу-лист
    tabu_list.append(best_neighbor_move)
    
    # 7. Обновляем лучшее известное решение
    if current_penalty < best_penalty:
        best_solution = current_solution
        best_penalty = current_penalty
        print(f"Итерация {i}: Найден новый лучший штраф: {best_penalty}")

# 8. Вывод результата
print("\n--- Поиск завершен ---")
print(f"Финальная (лучшая) оценка качества (штраф): {best_penalty}")
print_schedule(best_solution, "Финальное оптимальное расписание")

if best_penalty == 0:
    print("Статус: Оптимальное решение найдено (все ограничения выполнены).")
else:
    print(f"Статус: Найдено локально-оптимальное решение. Осталось штрафа: {best_penalty}.")

#######################################################################################################################################

# Вывод в консоль:

Запускаем Табу-поиск...

--- Начальное случайное расписание ---
       ПН ВТ СР ЧТ ПТ СБ ВС
       -------------------
Сотр 1: В  Р  В  Р  Р  Р  Р
Сотр 2: Р  Р  В  Р  Р  В  Р
Сотр 3: Р  Р  Р  Р  В  Р  В
Сотр 4: Р  Р  Р  Р  Р  В  В
Сотр 5: В  Р  В  Р  Р  Р  Р
-------------------------
Начальный штраф: 0

Идеальное решение найдено на итерации 0!

--- Поиск завершен ---
Финальная (лучшая) оценка качества (штраф): 0

--- Финальное оптимальное расписание ---
       ПН ВТ СР ЧТ ПТ СБ ВС
       -------------------
Сотр 1: В  Р  В  Р  Р  Р  Р
Сотр 2: Р  Р  В  Р  Р  В  Р
Сотр 3: Р  Р  Р  Р  В  Р  В
Сотр 4: Р  Р  Р  Р  Р  В  В
Сотр 5: В  Р  В  Р  Р  Р  Р
-------------------------
Статус: Оптимальное решение найдено (все ограничения выполнены).