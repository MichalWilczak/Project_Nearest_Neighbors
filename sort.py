import numpy as np

# Funkcja do sortowania w kolejności rosnącej

def sort(lista1):
    """
    Funkcja do sortowania w kolejności rosnącej
    Parametry:
    lista1: lista do posortowania
    Zwraca:
    np.array: posortowana lista typu np.array
    """
    lista1 = np.array(lista1)
    lista = lista1.copy().tolist()
    for a in range(len(lista)):
        for z in range(1, len(lista)):
            if lista[z] < lista[z-1]:
                bufor = lista[z]
                lista[z] = lista[z-1]
                lista[z-1] = bufor
    return np.array(lista)