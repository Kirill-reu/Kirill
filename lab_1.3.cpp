// Пример кода для реализации бинарной кучи на C++
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std;
template <typename T> class MinHeap {
    private:
    vector<T> heap;
    // Get the parent index
    int parent(int index) { return (index - 1) / 2; }
    // Get the left child index
    int leftChild(int index) { return (2* index + 1); }
    // Get the right child index
    int rightChild(int index) { return (2* index + 2); }
    // Heapify up to maintain heap property
    void heapifyUp(int index)
    {
        while (index && heap[index] < heap[parent(index)]) {
            swap(heap[index], heap[parent(index)]);
            index = parent(index);
        }
    }
    // Heapify down to maintain heap property
    void heapifyDown(int index)
    {
        int left = leftChild(index);
        int right = rightChild(index);
        int smallest = index;
        if (left < heap.size()
                && heap[left] < heap[smallest])
            6
        smallest = left;
        if (right < heap.size()
                && heap[right] < heap[smallest])
            smallest = right;
        if (smallest != index) {
            swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }
    public:
    // Check if the heap is empty
    bool isEmpty() const { return heap.empty(); }
    // Get the size of the heap
    int size() const { return heap.size(); }
    // Get the minimum element
    T getMin() const
    {
        if (isEmpty())
            throw runtime_error("Heap is empty");
        return heap.front();
    }
    // Insert a new key
    void insert(T key)
    {
        heap.push_back(key);
        int index = size() - 1;
        heapifyUp(index);
    }
    // Extract the minimum element
    T extractMin()
    {
        if (isEmpty())
            throw runtime_error("Heap is empty");
        T root = heap.front();
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
        return root;
    }
    // Delete a specific node
    void deleteNode(T key)
    {
        int index = -1;
// Find the index of the node to delete
        for (int i = 0; i < size(); ++i) {
            if (heap[i] == key) {
                index = i;
                break;
            }
        }
        if (index == -1)
            throw runtime_error(
                    7
                    "Key not found in the heap");
// Replace the node with the last element
        heap[index] = heap.back();
        heap.pop_back();
// Restore heap property
        heapifyUp(index);
        heapifyDown(index);
    }
    // Decrease key function
    void decreaseKey(int i, T newValue)
    {
        if (i < 0 || i >= size() || newValue > heap[i])
            throw invalid_argument(
                    "Invalid index or new value");
        heap[i] = newValue;
        heapifyUp(i);
    }
    // Print the heap elements
    void printHeap() const
    {
        for (const T& elem : heap)
        cout << elem << " ";
        cout << endl;
    }
};
int main()
{
    MinHeap<int> minHeap;
    minHeap.insert(3);
    minHeap.insert(2);
    minHeap.insert(15);
    minHeap.insert(5);
    minHeap.insert(4);
    minHeap.insert(45);
    cout << "Min Heap: ";
    minHeap.printHeap();
    cout << "Minimum element: " << minHeap.getMin() << endl;
    minHeap.extractMin();
    cout << "After extracting min: ";
    minHeap.printHeap();
// Delete node from min-heap
    minHeap.deleteNode(4);
    cout << "After deleting 4: ";
    minHeap.printHeap();
// Decrease key in min-heap
    minHeap.decreaseKey(2, 1);
    cout << "After decreasing key at index 2 to 1: ";
    minHeap.printHeap();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример кода для реализации биномиальной кучи на C++
#include <iostream>
#include <cstdlib>
using namespace std;
/*
 * Node Declaration
 */
struct node
{
int n;
int degree;
node* parent;
node* child;
node* sibling;
};
/*
 * Class Declaration
 */
class BinomialHeap
{
    private:
    node*H;
    node*Hr;
    int count;
    public:
    node* Initializeheap();
    int Binomial_link(node*, node*);
    node* Create_node(int);
    node* Union(node*, node*);
    node* Insert(node*, node*);
    node* Merge(node*, node*);
    node* Extract_Min(node*);
    int Revert_list(node*);
    int Display(node*);
    node* Search(node*, int);
    int Decrease_key(node*, int, int);
    int Delete(node*, int);
    BinomialHeap()
    {
        H = Initializeheap();
        Hr = Initializeheap();
        int count = 1;
    }
};
/*
 * Initialize Heap
 */
node* BinomialHeap::Initializeheap()
{
    node* np;
    np = NULL;
    return np;
}
/*
 * Linking nodes in Binomial Heap
 */
int BinomialHeap::Binomial_link(node* y, node* z)
{
    y->parent = z;
    y->sibling = z->child;
    z->child = y;
    z->degree = z->degree + 1;
}
/*
 * Create Nodes in Binomial Heap
 */
node* BinomialHeap::Create_node(int k)
{
    node* p = new node;
    p->n = k;
    return p;
}
/*
 * Insert Nodes in Binomial Heap
 */
node* BinomialHeap::Insert(node* H, node* x)
{
    node* H1 = Initializeheap();
    x->parent = NULL;
    x->child = NULL;
    x->sibling = NULL;
    x->degree = 0;
    H1 = x;
    H = Union(H, H1);
    return H;
}
/*
 * Union Nodes in Binomial Heap
 */
node* BinomialHeap::Union(node* H1, node* H2)
{
    node*H = Initializeheap();
    H = Merge(H1, H2);
    if (H == NULL)
        return H;
    node* prev_x;
    node* next_x;
    node* x;
    prev_x = NULL;
    x = H;
    next_x = x->sibling;
    while (next_x != NULL)
    {
        if ((x->degree != next_x->degree) || ((next_x->sibling != NULL)
                && (next_x->sibling)->degree == x->degree))
        {
            prev_x = x;
            x = next_x;
        }
else
        {
            if (x->n <= next_x->n)
            {
                x->sibling = next_x->sibling;
                Binomial_link(next_x, x);
            }
            else
            {
                if (prev_x == NULL)
                    H = next_x;
                else
                    prev_x->sibling = next_x;
                Binomial_link(x, next_x);
                x = next_x;
            }
        }
        next_x = x->sibling;
    }
    return H;
}
/*
 * Merge Nodes in Binomial Heap
 */
node* BinomialHeap::Merge(node* H1, node* H2)
{
    node* H = Initializeheap();
    node* y;
    node* z;
    node* a;
    node* b;
    y = H1;
    z = H2;
    if (y != NULL)
    {
        if (z != NULL)
        {
            if (y->degree <= z->degree)
                H = y;
            else if (y->degree > z->degree)
                H = z;
        }
        else
            H = y;
    }
    else
        H = z;
    while (y != NULL && z != NULL)
    {
        if (y->degree < z->degree)
        {
            y = y->sibling;
        }
        else if (y->degree == z->degree)
        {
            a = y->sibling;
            y->sibling = z;
            y = a;
        }
        else
        {
            b = z->sibling;
            z->sibling = y;
            z = b;
        }
    }
    return H;
}
/*
 * Display Binomial Heap
 */
int BinomialHeap::Display(node* H)
{
    if (H == NULL)
    {
        cout<<"The Heap is empty"<<endl;
        return 0;
    }
    cout<<"The root nodes are: "<<endl;
    node* p;
    p = H;
    while (p != NULL)
    {
        cout<<p->n;
        if (p->sibling != NULL)
            cout<<"-->";
        p = p->sibling;
    }
    cout<<endl;
}
/*
 * Extract Minimum
 */
node* BinomialHeap::Extract_Min(node* H1)
{
    Hr = NULL;
    node* t = NULL;
    node* x = H1;
    if (x == NULL)
    {
        cout<<"Nothing to Extract"<<endl;
        return x;
    }
    int min = x->n;
    node* p = x;
    while (p->sibling != NULL)
    {
        if ((p->sibling)->n < min)
        {
            min = (p->sibling)->n;
            t = p;
            x = p->sibling;
        }
        p = p->sibling;
    }
    if (t == NULL && x->sibling == NULL)
        H1 = NULL;
    else if (t == NULL)
        H1 = x->sibling;
    else if (t->sibling == NULL)
        t = NULL;
    else
        t->sibling = x->sibling;
    if (x->child != NULL)
    {
        Revert_list(x->child);
        (x->child)->sibling = NULL;
    }
    H = Union(H1, Hr);
    return x;
}
/*
 * Reverse List
 */
int BinomialHeap::Revert_list(node* y)
{
    if (y->sibling != NULL)
    {
        Revert_list(y->sibling);
        (y->sibling)->sibling = y;
    }
    else
    {
        Hr = y;
    }
}
/*
 * Search Nodes in Binomial Heap
 */
node* BinomialHeap::Search(node* H, int k)
{
    node* x = H;
    node* p = NULL;
    if (x->n == k)
    {
        p = x;
        return p;
    }
    if (x->child != NULL && p == NULL)
        p = Search(x->child, k);
    if (x->sibling != NULL && p == NULL)
        p = Search(x->sibling, k);
    return p;
}
/*
 * Decrease key of a node
 */
int BinomialHeap::Decrease_key(node* H, int i, int k)
{
    int temp;
    node* p;
    node* y;
    node* z;
    p = Search(H, i);
    if (p == NULL)
    {
        cout<<"Invalid choice of key"<<endl;
        return 0;
    }
    if (k > p->n)
    {
        cout<<"Error!! New key is greater than current key"<<endl;
        return 0;
    }
    p->n = k;
    y = p;
    z = p->parent;
    while (z != NULL && y->n < z->n)
    {
        temp = y->n;
        y->n = z->n;
        z->n = temp;
        y = z;
        z = z->parent;
    }
    cout<<"Key reduced successfully"<<endl;
}
/*
 * Delete Nodes in Binomial Heap
 */
int BinomialHeap::Delete(node* H, int k)
{
    node* np;
    if (H == NULL)
    {
        cout<<"\nHEAP EMPTY!!!!!";
        return 0;
    }
    Decrease_key(H, k, -1000);
    np = Extract_Min(H);
    if (np != NULL)
        cout<<"Node Deleted Successfully"<<endl;
}
/*
 * Main Contains Menu
 */
int main()
{
    int n, m, l, i;
    BinomialHeap bh;
    node* p;
    node*H;
    H = bh.Initializeheap();
    char ch;
    while (1)
    {
        cout<<"----------------------------"<<endl;
        cout<<"Operations on Binomial heap"<<endl;
        cout<<"----------------------------"<<endl;
        cout<<"1)Insert Element in the heap"<<endl;
        cout<<"2)Extract Minimum key node"<<endl;
        cout<<"3)Decrease key of a node"<<endl;
        cout<<"4)Delete a node"<<endl;
        cout<<"5)Display Heap"<<endl;
        cout<<"6)Exit"<<endl;
        cout<<"Enter Your Choice: ";
        cin>>l;
        switch(l)
        {
            case 1:
                cout<<"Enter the element to be inserted: ";
                cin>>m;
                p = bh.Create_node(m);
                H = bh.Insert(H, p);
                break;
            case 2:
                p = bh.Extract_Min(H);
                if (p != NULL)
                    cout<<"The node with minimum key: "<<p->n<<endl;
                else
                    cout<<"Heap is empty"<<endl;
                break;
            case 3:
                cout<<"Enter the key to be decreased: ";
                cin>>m;
                cout<<"Enter new key value: ";
                cin>>l;
                bh.Decrease_key(H, m, l);
                break;
            case 4:
                cout<<"Enter the key to be deleted: ";
                cin>>m;
                bh.Delete(H, m);
                break;
            case 5:
                cout<<"The Heap is: "<<endl;
                bh.Display(H);
                break;
            case 6:
                exit(1);
            default:
                cout<<"Wrong Choice";
        }
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример кода с реализацией «кучи Фибоначчи» на C++
#ifndef FIBONACCI_HEAP_HPP
#define FIBONACCI_HEAP_HPP
#include <vector>
#include <list>
#include <math.h>
typedef long long int lli;
#define GOLDEN_RATIO_ROUND_DOWN 1.618
template <class V> class FibonacciHeap;
template<class T>
class Element {
    protected:
    Element<T> *left = this;
    Element<T> *right = this;
    Element<T> *parent = nullptr;
    Element<T> *child = nullptr;
    T key;
    lli degree = 0;
    bool mark = false;
    public:
    Element(T key) : key(key) {};
    Element(Element<T> *other) : key(other->key), degree(other->degree), mark(other->mark) {
        this->left = other->left;
        this->right = other->right;
        this->parent = other->parent;
        this->child = other->child;
    }
    T getKey() { return this->key; }
    Element<T> *getLeft() { return this->left; }
    Element<T> *getRight() { return this->right; }
    Element<T> *getChild() { return this->child; }
    Element<T> *getParent() { return this->parent; }
    bool isMarked() { return this->mark; }
    lli getDegree() { return this->degree; }
    friend class FibonacciHeap<T>;
};
template<class T>
class FibonacciHeap {
    private:
    // Copy constructor and assignment operator are not implemented
// Hiding them to avoid misusage (Rule of three)
// Since the point here is show the algorithmic part of the data structure
    FibonacciHeap(const FibonacciHeap<T> &other);
    FibonacciHeap<T> &operator=(const FibonacciHeap<T> &other);
    protected:
    Element<T> *min;
    lli n;
    lli _D(lli n) {
        return log(n)/log(GOLDEN_RATIO_ROUND_DOWN);
    }
    void _deleteAll(Element<T> *x) {
        if (x != nullptr) {
            Element<T> *y = x;
            do {
                Element<T> *z = y;
                y = y->right;
                this->_deleteAll(z->child);
                delete z;
            } while (y != x);
        }
    }
    Element<T> *_unite(Element<T> *x, Element<T> *y) {
        if (x == nullptr) {
            return y;
        } else if (y == nullptr) {
            return x;
        } else if (x->key > y->key) {
            return this->_unite(y, x);
        } else {
            Element<T> *xRight = x->left;
            Element<T> *yRight = y->left;
            x->left = yRight;
            xRight->right = y;
            y->left = xRight;
            yRight->right = x;
            return x;
        }
    }
    void _link(Element<T> *y, Element<T> *x) {
        y->left->right = y->right;
        y->right->left = y->left;
        if (x->child == nullptr) {
            x->child = y;
            y->right = y;
            y->left = y;
        } else {
            Element<T> *child = x->child;
            y->right = child;
            y->left = child->left;
            child->left->right = y;
            child->left = y;
        }
        y->parent = x;
        x->degree++;
        y->mark = false;
    }
    void _fillListWithElements(Element<T> *x, std::list<Element<T>*> &A) {
        Element<T> *last = x;
        Element<T> *w = last;
        do {
            w = w->right;
            A.push_back(w);
        } while (w != last);
    }
    void _consolidate() {
        lli D = this->_D(this->n);
        std::vector<Element<T>*> A(D + 1, nullptr);
        std::list<Element<T>*> elements;
        this->_fillListWithElements(this->min, elements);
        for (auto x : elements) {
            lli d = x->degree;
            while (A.at(d) != nullptr) {
                Element<T> *y = A.at(d);
                if (x->key > y->key) {
                    std::swap(x, y);
                }
                this->_link(y, x);
                A.at(d) = nullptr;
                d++;
            }
            A.at(d) = x;
        }
        this->min = nullptr;
        for (lli i = 0; i < (lli)A.size(); i++) {
            if (A.at(i) != nullptr) {
                A.at(i)->right = A.at(i);
                A.at(i)->left = A.at(i);
                this->min = this->_unite(this->min, A.at(i));
            }
        }
    }
    void _cut(Element<T> *x, Element<T> *y) {
        x->left->right = x->right;
        x->right->left = x->left;
        if (y->child == x) {
            if (x->right == x) {
                y->child = nullptr;
            } else {
                y->child = x->right;
            }
        }
        y->degree--;
        x->left = x;
        x->right = x;
        x->parent = nullptr;
        x->mark = false;
        this->min = this->_unite(this->min, x);
    }
    void _cascadingCut(Element<T> *y) {
        Element<T> *z = y->parent;
        if (z != nullptr) {
            if (!y->mark) {
                y->mark = true;
            } else {
                this->_cut(y, z);
                this->_cascadingCut(z);
            }
        }
    }
    public:
    FibonacciHeap() {
    }
    this->min = nullptr;
this->n = 0;
};
~FibonacciHeap() {
    if (this->min != nullptr) {
        this->_deleteAll(this->min);
    }
}
bool isEmpty() const {
        return this->min == nullptr;
}
Element<T> *insert(T key) {
    Element<T> *x = new Element<T>(key);
    this->min = this->_unite(this->min, x);
    this->n++;
    return x;
}
void unite(FibonacciHeap<T> *heap) {
    this->min = this->_unite(this->min, heap->min);
    this->n += heap->n;
    heap->min = nullptr;
    heap->n = 0;
}
T getMin() const {
        return this->min->key;
}
T extractMin() {
    Element<T> *z = this->min;
    if (z == nullptr) {
        throw std::invalid_argument("heap is empty");
    }
    Element<T> *x = z->child;
    if (x != nullptr) {
        Element<T> *last = x;
        do {
            x = x->right;
            x->parent = nullptr;
        } while (x != last);
        this->min = this->_unite(this->min, x);
    }
    z->left->right = z->right;
    z->right->left = z->left;
    if (z == z->right) {
        this->min = nullptr;
    } else {
        this->min = z->right;
        this->_consolidate();
    }
    this->n--;
    T min = z->key;
    delete z;
    return min;
}
void decreaseKey(Element<T>* x, T k) {
    if (k > x->key) {
        throw std::invalid_argument("new key is greater than current key");
    }
    x->key = k;
    Element<T> *y = x->parent;
    if (y != nullptr && x->key < y->key) {
        this->_cut(x, y);
        this->_cascadingCut(y);
    }
    if (x->key < this->min->key) {
        this->min = x;
    }
}
void deleteElement(Element<T> *x) {
    this->decreaseKey(x, std::numeric_limits<T>::min());
    this->extractMin();
} 
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Пример с использованием std::unordered_map (std::unordered_map — контейнер, который реализует 
хэш-таблицу, хранящую пары «ключ-значение»)
#include <iostream> #include <unordered_map> using namespace std;
int main() {
// Создать хэш-таблицу 
    unordered_map<string, int> umap;
// Вставить пары «ключ-значение» 
    umap["Apple"] = 10;
    umap["Mango"] = 20;
    umap["Cherry"] = 30;
// Распечатать пары «ключ-значение» 
    for (auto it : umap) cout << it.first << " " << it.second << endl;
    return 0;
} 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Реализация хеш-таблицы с использованием вручную на основе массива связанных списков: 
#include <iostream>
#include <list>
#include <functional>
        // Структура для хранения пары ключ-значение
        template<typename K, typename V>
struct HashNode {
K key;
V value;
HashNode(K k, V v) : key(k), value(v) {}
        };
        // Класс хеш-таблицы
        template<typename K, typename V>
class HashTable {
    private:
    std::list<HashNode<K, V>>* table;
    size_t capacity;
    size_t size;
    // Простая хеш-функция
    size_t hashFunction(const K& key) {
        return std::hash<K>()(key) % capacity;
    }
    public:
    HashTable(size_t cap = 10) : capacity(cap), size(0) {
        table = new std::list<HashNode<K, V>>[capacity];
    } 
    ~HashTable() {
        delete[] table;
    }
    // Вставка элемента 
    void insert(const K& key, const V& value) {
        size_t index = hashFunction(key);

        // Проверяем, существует ли уже такой ключ 
        for (auto& node : table[index]) {
            if (node.key == key) {
                node.value = value; // Обновляем значение 
                return;
            }
        }
        // Добавляем новую пару ключ-значение 
        table[index].push_back(HashNode<K, V>(key, value));
        size++;
    }
    // Поиск значения по ключу 
    V* find(const K& key) {
        size_t index = hashFunction(key);

        for (auto& node : table[index]) {
            if (node.key == key) {
                return &node.value;
            }
        }
        return nullptr;
    }
    // Удаление элемента 
    void remove(const K& key) {
        size_t index = hashFunction(key);

        for (auto it = table[index].begin(); it != table[index].end(); ++it) {
            if (it->key == key) {
                table[index].erase(it);
                size--;
                return;
            }
        }
    }
    // Получение размера таблицы 
    size_t getSize() const {
        return size;
    }
};
int main() {
    HashTable<std::string, int> hashTable;

    hashTable.insert("one", 1);
    hashTable.insert("two", 2);
    hashTable.insert("three", 3);
    std::cout << "Value for 'one': " << *(hashTable.find("one")) << std::endl;
    hashTable.remove("two");
    std::cout << "Size of hash table: " << hashTable.getSize() << std::endl;
    return 0;
}