import matplotlib.pyplot as plt
from kNN import kNN
from kNN_scikitlearn import kNN_scikitlearn
from compare_kNN_labels import compare_kNN_labels


# Funkcja do porównywania własnego algorytmu z algorytmem z scikit_learn dla różnych wartości k i wyświetlająca na wykresie
def compare_algorithms(x_train, y_train, x_test, labels):
    """
    Funkcja do porównywania własnego algorytmu z algorytmem z scikit_learn dla różnych wartości k i wyświetlająca na wykresie

    Parametry:
    x_train: tablica z danymi treningowymi (zbiór obrazów ze zbioru MNIST jako wektorów)
    y_train: tablica numpy z etykietami dla obrazów ze zbioru treningowego
    x_test: tablica z danymi testowymi (zbiór obrazów ze zbioru MNIST jako wektorów)
    labels: tablica numpy z etykietami całego zbioru MNIST
    """
    percentages = []
    percentages_scikitlearn = []
    k_range = range(1, 10)
    for k in k_range:
        predicted_labels = kNN(x_train, y_train, x_test, k)
        predicted_scikitlearn = kNN_scikitlearn(x_train, y_train, x_test, k)
        kNN_fidelity, kNN_percentage = compare_kNN_labels(labels, predicted_labels)
        scikitlearn_fidelity, scikitlearn_percentage = compare_kNN_labels(labels, predicted_scikitlearn)
        percentages.append(kNN_percentage)
        percentages_scikitlearn.append(scikitlearn_percentage)
    plt.plot(k_range, percentages, label="Procentowa zgodność działania własnej funkcji", color="green")
    plt.plot(k_range, percentages_scikitlearn, label="Procentowa zgodność działania funkcji bibliotecznej", color="red")
    plt.xlabel('k')
    plt.ylabel('Zgodność w %')
    plt.title('Procentowa zgodność działania algorytmów Nearest Neighbors')
    plt.grid(True)
    plt.legend()
    plt.show()