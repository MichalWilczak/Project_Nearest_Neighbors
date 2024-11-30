import numpy as np
from vector_distance import vector_distance
from sort import sort
from collections import Counter

# Funkcja do przypisywania klas (etykiet) do obrazów (danych wejściowych)
def kNN(trainX, trainY, testX, k=3):
    """
        Funkcja do przypisywania klas (etykiet) do obrazów (danych wejściowych).

        Parametry:
        trainX: Lista tablic numpy (obrazów jako wektory) ze zbiorem treningowym
        trainY: Tablica numpy z etykietami do obrazów ze zbioru treningowego
        testX: Tablica numpy z danymi testowymi (obrazami w postaci wektorów)

        Zwraca:
        np.array: Lista etykiet dla obrazów na wejściu przypisanych na podstawie algorytmu kNN
    """
    classes = []
    # Obliczanie odległości pomiędzy testowymi obrazami a wszystkimi obrazami w zbiorze treningowym
    for input_image in testX:
        distances = np.array([(vector_distance(input_image, train_image), label) for train_image, label in zip(trainX, trainY)])
        # Sortowanie na podstawie odległości w kolejności rosnącej
        k_nearest_neighbors = sort(distances)[:k]
        # Wybór najczęściej występującej klasy wśród k najbliższych sąsiadów
        # Wybór drugiego elementu z każdej krotki w k_nearest_neighbors (_ oznacza ignorowanie)
        k_nearest_labels = [label for _, label in k_nearest_neighbors]
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        # Dodanie tej klasy do wyników
        classes.append(int(most_common_label))
    return np.array(classes)