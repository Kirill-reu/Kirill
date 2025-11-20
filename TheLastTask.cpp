// Выполнение задания 9

#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

using namespace std;

int knapsackGreedy(vector<int> weights, vector<int> values, int W) {
    vector<pair<double, int>> items;
    
    for (int i = 0; i < weights.size(); i++) {
        items.push_back({(double)values[i] / weights[i], i});
    }
    
    sort(items.rbegin(), items.rend());
    
    int totalValue = 0;
    int totalWeight = 0;
    
    for (auto& p : items) {
        int i = p.second;
        
        if (totalWeight + weights[i] <= W) {
            totalWeight += weights[i];
            totalValue += values[i];
        }
    }
    
    return totalValue;
}

int main() {
    // Тестовые данные

    int n = 3, W = 50;
    vector<int> weights = {10, 20, 30};
    vector<int> values = {60, 100, 120};
    
    cout << "Введите количество предметов и вместимость рюкзака: ";
    cout << n << " " << W << endl;
    
    cout << "Введите вес и стоимость для каждого предмета:" << endl;
    for (int i = 0; i < n; i++) {
        cout << weights[i] << " " << values[i] << endl;
    }
    
    int result = knapsackGreedy(weights, values, W);
    
    cout << "Максимальная ценность (жадный метод): " << result << endl;
    
    return 0;
}

 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Входные данные:
1) Количество предметов: 3
2) Вместимость рюкзака: 50
3) Веса и стоимости: Предмет 1: вес=10, стоимость=60; Предмет 2: вес=20, стоимость=100; Предмет 3: вес=30, стоимость=120

Вывод программы:
Введите количество предметов и вместимость рюкзака: 3 50
Введите вес и стоимость для каждого предмета:
10 60
20 100
30 120
Максимальная ценность (жадный метод): 160