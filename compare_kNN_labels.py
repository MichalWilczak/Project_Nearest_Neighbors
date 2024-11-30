
# Funkcja do porównywania działania funkcji kNN ze wzorcem
def compare_kNN_labels(labels, predictions):
    """
    Funkcja do porównywania działania funkcji kNN ze wzorcem

    Parametry:
    labels, predictions: Tablice numpy z etykietami do porównania

    Zwraca:
    krotka(tuple): element 1: liczba etykiet zgodna z oryginałem, element 2: procent etykiet zgodnych z oryginałem
    """
    fidelity = 0
    for x in range(len(predictions)):
        if labels[x] == predictions[x]:
            fidelity = fidelity + 1
    percentage_of_fidelity = (fidelity/len(predictions))*100
    return fidelity, percentage_of_fidelity