Задание 9. Метод ветвей и границ (приближённый) 

Условие. Решить задачу о рюкзаке (0-1) приближённо: отсортировать предметы по удельной стоимости и жадно набирать. 

Алгоритм: жадная эвристика для рюкзака. 

Язык примера: C++

<int knapsackGreedy(vector<int> weights, vector<int> values, int W) { 
vector<pair<double, int>> items; // (удельная стоимость, индекс) 
for (int i = 0; i < weights.size(); i++) { 
items.push_back({(double)values[i]/weights[i], i}); 
} 
sort(items.rbegin(), items.rend()); 
int totalValue = 0, totalWeight = 0; 
for (auto& p : items) { 
int i = p.second; 
// ДОПИСАТЬ: если вес позволяет, взять предмет 
} 
return totalValue; 
}>
