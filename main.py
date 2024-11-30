import numpy as np
from sklearn.datasets import load_digits
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize
from kNN import kNN
from kNN_scikitlearn import kNN_scikitlearn
from compare_kNN_labels import compare_kNN_labels
from compare_algorithms import compare_algorithms
from compare_time import compare_time
import unittest
import numpy.testing as npt
from sort import sort
from vector_distance import vector_distance
from math import isclose



# Wczytanie zbiorów danych
digits8 = load_digits()
train_dataset28 = MNIST(root='mnist_data', train=True, download=True, transform=ToTensor())
dataset16 = MNIST(root='mnist_data', download=True, transform=Compose([Resize((16, 16)), ToTensor()]))


# Wczytanie etykiet dla kolejnych obrazów ze zbioru MNIST ze scikit-learn i PyTorch
labels8 = digits8.target
labels28 = np.array([train_dataset28[i][1] for i in range(len(train_dataset28))])
labels16 = np.array([dataset16[i][1] for i in range(len(dataset16))])
# Obrazy jako macierze
images8 = digits8.images
images28 = np.array([train_dataset28[i][0] for i in range(len(train_dataset28))])
images16 = np.array([dataset16[i][0] for i in range(len(dataset16))])
# Obrazy jako wektory
images_as_vectors8 = []
images_as_vectors28 = []
images_as_vectors16 = []
for image8 in images8:
    vector8 = image8.flatten()
    images_as_vectors8.append(vector8)
for image28 in images28:
    vector28 = image28.flatten()
    images_as_vectors28.append(vector28)
for image16 in images16:
    vector16 = image16.flatten()
    images_as_vectors16.append(vector16)

#################################################################################################################
# Test działania kNN
#################################################################################################################

# Dane treningowe (x_train) i ich etykiety (y_train)
first_image_from_dataset = 500
last_image_from_dataset = 1500
#1797
x_train = images_as_vectors8[first_image_from_dataset:last_image_from_dataset]
y_train = labels8[first_image_from_dataset:last_image_from_dataset]

x_train28 = images_as_vectors28[first_image_from_dataset:last_image_from_dataset]
y_train28 = labels28[first_image_from_dataset:last_image_from_dataset]

x_train16 = images_as_vectors16[first_image_from_dataset:last_image_from_dataset]
y_train16 = labels16[first_image_from_dataset:last_image_from_dataset]

first_test_image = 0
last_test_image = 500

# Dane testowe (x_test)
x_test = images_as_vectors8[first_test_image:last_test_image]
x_test28 = images_as_vectors28[first_test_image:last_test_image]
x_test16 = images_as_vectors16[first_test_image:last_test_image]

# Przewidywanie klasy dla danych testowych
k = 4
predictions = kNN(x_train, y_train, x_test, k)
predictions28 = kNN(x_train28, y_train28, x_test28, k)
predictions16 = kNN(x_train16, y_train16, x_test16, k)
predictions_scikitlearn = kNN_scikitlearn(x_train, y_train, x_test, k)
predictions_scikitlearn28 = kNN_scikitlearn(x_train28, y_train28, x_test28, k)
predictions_scikitlearn16 = kNN_scikitlearn(x_train16, y_train16, x_test16, k)

print(f"Etykiety analizowanych obrazów 8x8: \n{labels8[first_test_image:last_test_image]}")
print(f"Predykcje etykiet własnej funkcji kNN dla 8x8: \n{predictions}")
print(f"Predykcje etykiet funkcji kNN ze scikit-learn dla 8x8: \n{predictions_scikitlearn}")
fidelity, percentage_of_fidelity = compare_kNN_labels(labels8, predictions)
print(f"Porównanie własnej funkcji z oryginalnymi etykietami dla 8x8 i k = {k}:")
print(f"{fidelity} etykiet z {len(predictions)} jest zgodne z oryginałem, co stanowi {percentage_of_fidelity}%")
print(f"Porównanie funkcji z biblioteki scikit-learn z oryginalnymi etykietami dla 8x8 i k = {k}:")
fidelity, percentage_of_fidelity = compare_kNN_labels(labels8, predictions_scikitlearn)
print(f"{fidelity} etykiet z {len(predictions)} jest zgodne z oryginałem, co stanowi {percentage_of_fidelity}%\n")



print(f"Etykiety analizowanych obrazów 28x28: \n{labels28[first_test_image:last_test_image]}")
print(f"Predykcje etykiet własnej funkcji kNN dla 28x28: \n{predictions28}")
print(f"Predykcje etykiet funkcji kNN ze scikit-learn dla 28x28: \n{predictions_scikitlearn28}")
fidelity, percentage_of_fidelity = compare_kNN_labels(labels28, predictions28)
print(f"Porównanie własnej funkcji z oryginalnymi etykietami dla 28x28 i k = {k}:")
print(f"{fidelity} etykiet z {len(predictions28)} jest zgodne z oryginałem, co stanowi {percentage_of_fidelity}%")
print(f"Porównanie funkcji z biblioteki scikit-learn z oryginalnymi etykietami dla 28x28 i k = {k}:")
fidelity, percentage_of_fidelity = compare_kNN_labels(labels28, predictions_scikitlearn28)
print(f"{fidelity} etykiet z {len(predictions28)} jest zgodne z oryginałem, co stanowi {percentage_of_fidelity}%\n")



print(f"Etykiety analizowanych obrazów 16x16: \n{labels16[first_test_image:last_test_image]}")
print(f"Predykcje etykiet własnej funkcji kNN dla 16x16: \n{predictions16}")
print(f"Predykcje etykiet funkcji kNN ze scikit-learn dla 16x16: \n{predictions_scikitlearn16}")
fidelity, percentage_of_fidelity = compare_kNN_labels(labels16, predictions16)
print(f"Porównanie własnej funkcji z oryginalnymi etykietami dla 16x16 i k = {k}:")
print(f"{fidelity} etykiet z {len(predictions16)} jest zgodne z oryginałem, co stanowi {percentage_of_fidelity}%")
print(f"Porównanie funkcji z biblioteki scikit-learn z oryginalnymi etykietami dla 16x16 i k = {k}:")
fidelity, percentage_of_fidelity = compare_kNN_labels(labels16, predictions_scikitlearn16)
print(f"{fidelity} etykiet z {len(predictions16)} jest zgodne z oryginałem, co stanowi {percentage_of_fidelity}%\n")


# Porównanie algorytmów
compare_algorithms(x_train, y_train, x_test, labels8)
compare_algorithms(x_train28, y_train28, x_test28, labels28)
compare_algorithms(x_train16, y_train16, x_test16, labels16)

compare_time(x_train, y_train, x_test, x_train28, y_train28, x_test28, x_train16, y_train16, x_test16)

class TestSortFunction(unittest.TestCase):
    def test_sort_list(self):
        npt.assert_array_equal(sort([9, 1, 3, 8, 6, 5, 1]), np.array([1, 1, 3, 5, 6, 8, 9]))
    def test_empty(self):
        npt.assert_array_equal(sort([]), np.array([]))
    def test_one_element(self):
        npt.assert_array_equal(sort([10]), np.array([10]))
    def test_sorted(self):
        npt.assert_array_equal(sort([1, 2, 3, 4, 5, 6, 7, 8]), np.array([1, 2, 3, 4, 5, 6, 7, 8]))

class TestVectorDistance(unittest.TestCase):
    def test_different_vectors(self):
        wektor1 = np.zeros(64)
        wektor2 = np.ones(64)
        self.assertTrue(isclose(vector_distance(wektor1, wektor2), 8))
    def test_equal_vectors(self):
        wektor1 = np.ones(64)
        wektor2 = wektor1
        self.assertTrue(isclose(vector_distance(wektor1, wektor2), 0))


if __name__ == '__main__':
    unittest.main()

