// Создание мультисписка на языке С++
#include <iostream>
#include <vector>

struct Node {
  int data;
  std::vector<Node*> lists;
  Node(int val) : data(val) {}
};

int main() {
  Node *prev = new Node(1);
  Node *next = new Node(2);

  head->lists.push_back(second); // Добавляем второй список к первому

  std::cout << head->lists[0]->data << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////

// Создание очереди на языке C++
#include <iostream>
#include <queue>
#include <string>

int main() {
  std::queue<std::string> Queue;

  Queue.push("Tom"); // Добавление элементов в очередь
  Queue.push("Bob");
  Queue.push("Sam");

  std::cout << "Первый элемент: " << Queue.front() << std::endl; // Получение первого элемента
  std::cout << "Последний элемент: " << Queue.back() << std::endl; // Получение последнего элемента

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////

// Реализация структуры дека на языке C++
#include <deque>
#include <iostream> 

std::deque<int> d = {1, 2, 3}; 
d.push_front(0); 
d.push_back(4); 
for (int num : d) { std::cout << num << " "; } 

return 0; 

///////////////////////////////////////////////////////////////////////////////////////////////

// Создание структуры приоритетной очереди на языке C++
#include <iostream> 
#include <queue>

int main() { 
std::priority_queue<int> pq; 

pq.push(10); 
pq.push(20); 
pq.push(15);
 
std::cout << "The highest priority element is: " << pq.top() << std::endl; 
pq.pop(); 
std::cout << "After popping, the highest priority element is: " << pq.top() << std::endl; 

return 0; 
}