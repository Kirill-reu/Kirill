// Пример структуры для реализации бинарной кучи на Java

// Явное управление размером массива
public class MinHeap {
    private int[] heap;
    private int size;
    
    public MinHeap(int capacity) {
        heap = new int[capacity];  // Фиксированный размер изначально
        size = 0;
    }
    
    private void siftUp(int idx) {
        int parent = (idx - 1) / 2;
        if (idx > 0 && heap[idx] < heap[parent]) {
            int temp = heap[idx];  // Явное копирование
            heap[idx] = heap[parent];
            heap[parent] = temp;
            siftUp(parent);
        }
    }
    
    public void push(int val) {
        if (size == heap.length) resize();  // Нужно самому расширять
        heap[size] = val;
        siftUp(size++);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример структуры для реализации биноминальной кучи на Java

class BinomialNode<T extends Comparable<T>> {
    T key;  // Дженерик тип
    int degree;
    BinomialNode<T> child;    // Типизированные ссылки
    BinomialNode<T> sibling;  // JVM управляет памятью
    
    public BinomialNode(T key) {
        this.key = key;
    }
}

class BinomialHeap<T extends Comparable<T>> {
    private List<BinomialNode<T>> trees;  // Типизированная коллекция
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример структуры для реализации «кучи Фибоначчи» на Java

class FibonacciNode<T extends Comparable<T>> {
    T key;
    int degree;
    boolean marked;
    FibonacciNode<T> parent, child;
    FibonacciNode<T> left, right;  // Циклические ссылки
    
    // JVM справится с циклическими ссылками (modern GC)
}

class FibonacciHeap<T extends Comparable<T>> {
    private FibonacciNode<T> minNode;
    private int count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример структуры с реализацией «Хэш-таблицы» на Java

class HashMap<K, V> {
    static class Entry<K, V> {
        final K key;  // final для иммутабельности
        V value;
        Entry<K, V> next;  // Связный список для цепочек
        
        Entry(K k, V v) { key = k; value = v; }
    }
    
    private Entry<K, V>[] table;
    private int capacity;
    
    public V put(K key, V value) {
        int hash = key.hashCode() % capacity;  // Использует hashCode()
        // ... разрешение коллизий через цепочки
    }
}
