#Создание мультисписков на языке python
groups = [['Hong', 'Ryan'], ['Andry', 'Ross'], ['Mike', 'Smith']]
names = []
for group in groups:
    for name in group:
        names.append(name)
print(names)

#Создание очереди на языке python
from queue import Queue 
q = Queue() 
q.put(1) 
q.put(2) 
q.put(3) 
while not q.empty(): 
    print(q.get())

#Реализация дека на языке python
from collections import deque 
tasks = deque() 
tasks.append("task1") 
tasks.append("task2") 
tasks.append("task3") 
while tasks: 
    current_task = tasks.popleft() 
    print(f"Выполняется: {current_task}")

#Создание приоритетной очереди на языке python
from queue import PriorityQueue 
q = PriorityQueue() 
q.put((2, 'mid-priority item')) 
q.put((1, 'high-priority item')) 
q.put((3, 'low-priority item')) 
while not q.empty(): 
    item = q.get()
    print(item)


