// Алгоритмы реализации графов на C++

// 1. Представление графа (список смежности)

struct Edge {
    int destination;
    int weight;
};

vector<vector<Edge>> graph(8);

// Добавление рёбер
graph[0] = {{1, 2}, {2, 5}};
graph[1] = {{2, 2}, {3, 3}, {4, 8}};
// ... остальные вершины

////////////////////////////////////////////////////////////////////////////////////////

// 2. Алгоритм Дейкстры

void dijkstra(const vector<vector<Edge>>& graph, int source, int destination) {
    int n = graph.size();
    vector<int> distances(n, numeric_limits<int>::max());
    vector<int> previous(n, -1);
    distances[source] = 0;
    
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int current = pq.top().second;
        int current_distance = pq.top().first;
        pq.pop();
        
        if (current_distance > distances[current]) continue;
        
        for (const Edge& edge : graph[current]) {
            int next = edge.destination;
            int weight = edge.weight;
            int distance = current_distance + weight;
            
            if (distance < distances[next]) {
                distances[next] = distance;
                previous[next] = current;
                pq.push({distance, next});
            }
        }
    }
    
    // Восстановление пути
    if (distances[destination] != numeric_limits<int>::max()) {
        vector<int> path;
        for (int at = destination; at != -1; at = previous[at]) {
            path.push_back(at);
        }
        reverse(path.begin(), path.end());
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// 3. Поиск пути в дереве

vector<int> getPathFromRootToNode(BTree* node, vector<int> v, int key) {
    if (node == NULL) {
        return v;
    }
    v.push_back(node->data);
    if (node->data == key) {
        return v;
    }
    if (v.back() != key) {
        v = getPathFromRootToNode(node->left, v, key);
    }
    if (v.back() != key) {
        v = getPathFromRootToNode(node->right, v, key);
    }
    if (v.back() != key) {
        v.pop_back();
    }
    return v;
}