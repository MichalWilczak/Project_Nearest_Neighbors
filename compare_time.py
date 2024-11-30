from measure_time import measure_time
import matplotlib.pyplot as plt


def compare_time(x_train, y_train, x_test, x_train28, y_train28, x_test28, x_train16, y_train16, x_test16):
    """
    Funkcja do wyświetlania na wykresie czasów wykonywania algorytmu kNN dla różnych rozmiarów obrazów MNIST i różnych k

    Parametry:
    x_train: tablica z danymi treningowymi (zbiór obrazów ze zbioru MNIST jako wektorów o długości 64)
    y_train: tablica numpy z etykietami dla obrazów ze zbioru treningowego
    x_test: tablica z danymi testowymi (zbiór obrazów ze zbioru MNIST jako wektorów o długości 64)
    x_train28: tablica z danymi treningowymi (zbiór obrazów ze zbioru MNIST jako wektorów o długości 784)
    y_train28: tablica numpy z etykietami dla obrazów ze zbioru treningowego
    x_test28: tablica z danymi testowymi (zbiór obrazów ze zbioru MNIST jako wektorów o długości 784)
    x_train16: tablica z danymi treningowymi (zbiór obrazów ze zbioru MNIST jako wektorów o długości 256)
    y_train16: tablica numpy z etykietami dla obrazów ze zbioru treningowego
    x_test16: tablica z danymi testowymi (zbiór obrazów ze zbioru MNIST jako wektorów o długości 256)
    """
    times_8x8 = measure_time(x_train, y_train, x_test)
    times_28x28 = measure_time(x_train28, y_train28, x_test28)
    times_16x16 = measure_time(x_train16, y_train16, x_test16)

    k_range = range(1, 10)

    plt.plot(k_range, times_8x8, label="Czasy wykonywania kNN dla obrazów 8x8", color="green")
    plt.plot(k_range, times_28x28, label="Czasy wykonywania kNN dla obrazów 28x28", color="red")
    plt.plot(k_range, times_16x16, label="Czasy wykonywania kNN dla obrazów 16x16", color="blue")
    plt.xlabel('k')
    plt.ylabel('Czas [s]')
    plt.title('Czasy wykonywania algorytmu kNN dla obrazów o różnych rozmiarach')
    plt.grid(True)
    plt.legend()
    plt.show()