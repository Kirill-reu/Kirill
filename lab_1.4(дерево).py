# Алгоритмы реализации деревьев на языке Python

# 1. Базовая структура узла

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class Tree:
    def __init__(self):
        self.root = None

########################################################################

# 2. Вставка элемента

# Рекурсивная реализация:
def insert_recursively(v, x):
    if v is None:
        return Node(x)
    if x < v.key:
        v.left = insert_recursively(v.left, x)
    elif x > v.key:
        v.right = insert_recursively(v.right, x)
    return v

# Использование:
tree.root = insert_recursively(tree.root, x)

# Нерекурсивная реализация:
def insert_iteratively(x):
    parent = None
    v = tree.root
    while v is not None:
        parent = v
        if x < v.key:
            v = v.left
        elif x > v.key:
            v = v.right
        else:
            return
    
    new_node = Node(x)
    if parent is None:
        tree.root = new_node
    elif x < parent.key:
        parent.left = new_node
    else:
        parent.right = new_node

########################################################################

# 3. Поиск элемента

# Рекурсивная реализация:
def search_recursively(v, x):
    if v is None or v.key == x:
        return v
    elif x < v.key:
        return search_recursively(v.left, x)
    else:
        return search_recursively(v.right, x)

# Нерекурсивная реализация:
def search_iteratively(x):
    v = root
    while v is not None:
        if v.key == x:
            return v
        elif x < v.key:
            v = v.left
        else:
            v = v.right
    return None

########################################################################

# 4. Удаление элемента

def delete_recursively(v, x):
    if v is None:
        return None
    
    if x < v.key:
        v.left = delete_recursively(v.left, x)
        return v
    elif x > v.key:
        v.right = delete_recursively(v.right, x)
        return v
    
    # v.key == x
    if v.left is None:
        return v.right
    elif v.right is None:
        return v.left
    else:
        # Оба поддерева присутствуют
        min_key = find_min(v.right).key
        v.key = min_key
        v.right = delete_recursively(v.right, min_key)
        return v

def find_min(v):
    if v.left is not None:
        return find_min(v.left)
    return v