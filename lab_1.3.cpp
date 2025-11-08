// Пример структуры для реализации бинарной кучи на C++

// Шаблонный класс с управлением памятью
template<typename T>
class MinHeap {
    std::vector<T> heap;
    
    void siftUp(int idx) {
        int parent = (idx - 1) / 2;
        if (idx > 0 && heap[idx] < heap[parent]) {  // Использует operator<
            std::swap(heap[idx], heap[parent]);  // STL функция
            siftUp(parent);
        }
    }
    
public:
    void push(const T& val) {
        heap.push_back(val);  // vector сам управляет памятью
        siftUp(heap.size() - 1);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример структуры для реализации биномиальной кучи на C++

template<typename T>
struct BinomialNode {
    T key;
    int degree;
    std::shared_ptr<BinomialNode<T>> child;    // Умные указатели!
    std::shared_ptr<BinomialNode<T>> sibling;  // Для автоматического управления
    std::weak_ptr<BinomialNode<T>> parent;     // weak_ptr для избежания циклов
    
    BinomialNode(const T& k) : key(k) {}
};

template<typename T>
class BinomialHeap {
    std::vector<std::shared_ptr<BinomialNode<T>>> trees;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример структуры с реализацией «кучи Фибоначчи» на C++

template<typename T>
class FibonacciNode {
public:
    T key;
    int degree;
    bool marked;
    std::shared_ptr<FibonacciNode> parent;
    std::shared_ptr<FibonacciNode> child;
    std::weak_ptr<FibonacciNode> left, right;  // weak_ptr для циклических ссылок!
    
    FibonacciNode(const T& k) : key(k) {
        left = right = shared_from_this();  // Сложная логика владения
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример структуры с реализацией «Хэш-таблицы» на C++

template<typename K, typename V>
class HashMap {
    struct Node {
        K key;
        V value;
        std::unique_ptr<Node> next;  // Уникальный указатель для владения
        
        Node(const K& k, const V& v) : key(k), value(v) {}
    };
    
    std::vector<std::unique_ptr<Node>> table;
    
    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % table.size();  // std::hash
    }
};
