// Алгоритмы реализации деревьев на Java

// 1. Базовая структура узла

class Tree {
    static class Node {
        int key;
        Node left;
        Node right;
        
        public Node(int key) {
            this.key = key;
        }
    }
    
    private Node root;
}

/////////////////////////////////////////////////////////////////////////////////////

// 2. Вставка элемента

public void insert(int x) {
    root = doInsert(root, x);
}

private Node doInsert(Node node, int x) {
    if (node == null) {
        return new Node(x);
    }
    if (x < node.key) {
        node.left = doInsert(node.left, x);
    } else if (x > node.key) {
        node.right = doInsert(node.right, x);
    }
    return node;
}
