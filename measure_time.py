import time
from kNN import kNN

def measure_time(x_train, y_train, x_test):
    """
    Funkcja do pomiaru czasu wykonywania algorytmu kNN dla różnych wartości k dla obrazów MNIST

    Parametry:
    x_train: tablica z danymi treningowymi (zbiór obrazów ze zbioru MNIST jako wektorów)
    y_train: tablica numpy z etykietami dla obrazów ze zbioru treningowego
    x_test: tablica z danymi testowymi (zbiór obrazów ze zbioru MNIST jako wektorów)

    Zwraca:
    measured_times: lista z czasami wykonywania funkcji kNN dla kolejnych parametrów k
    """
    k_range = range(1, 10)
    measured_times = []
    for k in k_range:
        start = time.perf_counter()
        kNN(x_train, y_train, x_test, k)
        end = time.perf_counter()
        elapsed_time = end - start
        measured_times.append(elapsed_time)
    return measured_times

