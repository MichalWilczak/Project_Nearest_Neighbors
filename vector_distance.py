import numpy as np

# Funkcja do obliczania odległości pomiędzy wektorami
def vector_distance(wektor1, wektor2):
    """
    Funkcja do obliczania odległości euklidesowej między dwoma wektorami.

    Parametry:
    wektor1, wektor2: Tablice numpy, reprezentujące dwa obrazy.

    Zwraca:
    float: Odległość euklidesowa między dwoma wektorami.
    """
    return np.sqrt(np.sum((wektor1 - wektor2) ** 2))