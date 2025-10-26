// Алгоритм сортировки выбором

public class SelectionSort {
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Алгоритм сортировки слиянием (Merge Sort)

import java.util.Arrays;

public class MergeSort {

    void merge(int arr[], int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;

        int L[] = new int[n1];
        int R[] = new int[n2];

        for (int i = 0; i < n1; ++i)
            L[i] = arr[l + i];
        for (int j = 0; j < n2; ++j)
            R[j] = arr[m + 1 + j];

        int i = 0, j = 0;
        int k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            } else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }

        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }

    void sort(int arr[], int l, int r) {
        if (l < r) {
            int m = (l + r) / 2;

            sort(arr, l, m);
            sort(arr, m + 1, r);

            merge(arr, l, m, r);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Алгоритм пирамидальной сортировки

public class HeapSort {
    public static void sort(int[] arr) {
        int n = arr.length;

        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }

        // 2. Извлечение элементов из кучи по одному
        for (int i = n - 1; i > 0; i--) {
            // Перемещаем текущий корень (наибольший элемент) в конец массива.
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;

            // Вызываем heapify на уменьшенной куче.
            // Элемент arr[i] теперь находится на своем месте.
            heapify(arr, i, 0);
        }
    }
    static void heapify(int[] arr, int n, int i) {
        int largest = i; // Инициализируем наибольший элемент как корень
        int left = 2 * i + 1; // Левый потомок
        int right = 2 * i + 2; // Правый потомок

        // Если левый потомок больше корня
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }

        // Если правый потомок больше, чем наибольший на данный момент
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }

        // Если наибольший элемент не корень
        if (largest != i) {
            // Меняем местами arr[i] и arr[largest]
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;

            // Рекурсивно преобразуем в кучу затронутое поддерево
            heapify(arr, n, largest);
        }
    }

    // Пример использования
    public static void main(String[] args) {
        int[] arr = {12, 11, 13, 5, 6, 7};
        sort(arr);
        System.out.print("Отсортированный массив: ");
        for (int i = 0; i < arr.length; ++i) {
            System.out.print(arr[i] + (i < arr.length - 1 ? ", " : ""));
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Алгоритм последовательного (линейного) поиска

public class LinearSearch {
    public static int linearSearch(int[] arr, int target) {
        // Получаем длину массива
        int n = arr.length;

        // Итерируем по всем элементам массива, начиная с индекса 0
        for (int i = 0; i < n; i++) {
            // Сравниваем текущий элемент с искомым значением
            if (arr[i] == target) {
                // Если совпадение найдено, возвращаем текущий индекс
                return i;
            }
        }

        // Если цикл завершился (просмотрен весь массив) и совпадение не найдено,
        // возвращаем -1, чтобы указать на отсутствие элемента.
        return -1;
    }

    // Пример использования
    public static void main(String[] args) {
        int[] data = {15, 30, 8, 45, 12, 9};
        int target1 = 45;
        int target2 = 25;

        // Поиск target1
        int index1 = linearSearch(data, target1);
        if (index1 != -1) {
            System.out.println("Элемент " + target1 + " найден по индексу: " + index1); // Вывод: 3
        } else {
            System.out.println("Элемент " + target1 + " не найден.");
        }

        // Поиск target2
        int index2 = linearSearch(data, target2);
        if (index2 != -1) {
            System.out.println("Элемент " + target2 + " найден по индексу: " + index2);
        } else {
            System.out.println("Элемент " + target2 + " не найден."); // Вывод: Элемент 25 не найден.
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Алгоритм интерполирующего поиска

public class InterpolationSearch {
    public static int interpolationSearch(int[] arr, int target) {
        int low = 0; // Нижняя граница поиска
        int high = arr.length - 1; // Верхняя граница поиска

        // Поиск продолжается, пока границы не пересекутся и target находится 
        // в диапазоне значений между arr[low] и arr[high].
        while (low <= high && target >= arr[low] && target <= arr[high]) {

            // Если границы совпали, сразу проверяем элемент
            if (low == high) {
                if (arr[low] == target) {
                    return low;
                } else {
                    return -1; // Не найдено
                }
            }

            // Главная формула интерполяции:
            // Вычисляет наиболее вероятную позицию элемента 'target'.
            // Обратите внимание на приведение типов для корректной математики.
            long pos = low + (
                ((long)high - low) * (target - arr[low]) / (arr[high] - arr[low])
            );

            // Преобразуем long обратно в int для индексации
            int mid = (int) pos; 

            // ----------------------------------------------------
            // Проверка элемента в вычисленной позиции 'mid'
            // ----------------------------------------------------
            if (arr[mid] == target) {
                // Элемент найден
                return mid;
            }

            if (arr[mid] < target) {
                // target больше, чем arr[mid], сдвигаем нижнюю границу
                low = mid + 1;
            } else {
                // target меньше, чем arr[mid], сдвигаем верхнюю границу
                high = mid - 1;
            }
        }

        // Если цикл завершился (target не в диапазоне или границы пересеклись)
        return -1;
    }

    // Пример использования
    public static void main(String[] args) {
        // Важно: массив должен быть отсортирован и иметь равномерное распределение для оптимальной работы
        int[] data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        int target1 = 70;
        int target2 = 75;

        // Поиск target1
        int index1 = interpolationSearch(data, target1);
        System.out.println("Элемент " + target1 + " найден по индексу: " + index1); // Вывод: 6

        // Поиск target2
        int index2 = interpolationSearch(data, target2);
        System.out.println("Элемент " + target2 + " найден по индексу: " + index2); // Вывод: -1
    }
}