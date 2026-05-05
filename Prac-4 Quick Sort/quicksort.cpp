#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i+1], arr[high]);
    return i + 1;
}

void seqQuickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        seqQuickSort(arr, low, pi - 1);
        seqQuickSort(arr, pi + 1, high);
    }
}

void parQuickSort(vector<int>& arr, int low, int high, int depth = 0) {
    if (low < high) {
        int pi = partition(arr, low, high);
        if (depth <= 3) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parQuickSort(arr, low, pi - 1, depth + 1);
                #pragma omp section
                parQuickSort(arr, pi + 1, high, depth + 1);
            }
        } else {
            seqQuickSort(arr, low, pi - 1);
            seqQuickSort(arr, pi + 1, high);
        }
    }
}

int main() {
    vector<int> a1 = {9, 3, 7, 1, 5, 8, 2, 6, 4, 0};
    vector<int> a2 = a1;

    double t;

    t = omp_get_wtime();
    seqQuickSort(a1, 0, a1.size() - 1);
    cout << "Sequential Time : " << omp_get_wtime() - t << " sec" << endl;

    t = omp_get_wtime();
    parQuickSort(a2, 0, a2.size() - 1);
    cout << "Parallel Time   : " << omp_get_wtime() - t << " sec" << endl;

    cout << "Sorted: ";
    for (int x : a2) cout << x << " ";
    cout << endl;

    return 0;
}
// Compile: g++ -fopenmp P4_Quicksort.cpp -o quicksort
// Run:     ./quicksort