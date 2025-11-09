// Алгоритмы реализации графов на Java

// 1. Представление графа (список смежности)

class Edge {
    int source;
    int destination;
    int weight;
    
    public Edge(int source, int destination, int weight) {
        this.source = source;
        this.destination = destination;
        this.weight = weight;
    }
}

// Создание графа
ArrayList<ArrayList<Edge>> graph = new ArrayList<>();

// Инициализация для 7 вершин
for (int i = 0; i < 7; i++) {
    graph.add(new ArrayList<>());
}

// Добавление рёбер
graph.get(0).add(new Edge(0, 1, 14));
graph.get(0).add(new Edge(0, 3, 5));
// ... остальные рёбра

/////////////////////////////////////////////////////////////////////////////////////////////

// 2. Алгоритм Дейкстры

static class Pair implements Comparable<Pair> {
    int vertex;
    int wsf; // weight so far
    
    public Pair(int vertex, int wsf) {
        this.vertex = vertex;
        this.wsf = wsf;
    }
    
    public int compareTo(Pair o) {
        return this.wsf - o.wsf;
    }
}

public static void dijkstra(ArrayList<ArrayList<Edge>> graph, int src) {
    boolean[] visited = new boolean[graph.size()];
    PriorityQueue<Pair> pq = new PriorityQueue<>();
    pq.add(new Pair(src, 0));
    
    while (!pq.isEmpty()) {
        Pair top = pq.remove();
        
        if (visited[top.vertex]) continue;
        
        visited[top.vertex] = true;
        System.out.println("Vertex: " + top.vertex + " & Weight: " + top.wsf);
        
        for (Edge edge : graph.get(top.vertex)) {
            if (!visited[edge.destination]) {
                pq.add(new Pair(edge.destination, top.wsf + edge.weight));
            }
        }
    }
}