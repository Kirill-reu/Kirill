// Алгоритмы реализации деревьев на C++

// 1. Базовая структура узла

class TNode {
public:
    int Key;
    TNode* Left;
    TNode* Right;
    
    TNode(int key) : Key(key), Left(nullptr), Right(nullptr) {}
};

class TTree {
private:
    TNode* Root;
public:
    TTree() : Root(nullptr) {}
};

///////////////////////////////////////////////////////////////////////////////////////

// 2.1 Вставка элемента

TNode* InsertRecursively(TNode* node, int x) {
    if (node == nullptr) {
        return new TNode(x);
    }
    if (x < node->Key) {
        node->Left = InsertRecursively(node->Left, x);
    } else if (x > node->Key) {
        node->Right = InsertRecursively(node->Right, x);
    }
    return node;
}

////////////////////////////////////////////////////////////////////////////////////////

// 2.2 Вставка элемента (с использованием указателей на указатели)

void Insert(int x) {
    TNode** cur = &Root;
    while (*cur) {
        TNode& node = **cur;
        if (x < node.Key) {
            cur = &node.Left;
        } else if (x > node.Key) {
            cur = &node.Right;
        } else {
            return;
        }
    }
    *cur = new TNode(x);
}