from sklearn.neighbors import KNeighborsClassifier

# Funkcja z biblioteki scikit-learn
def kNN_scikitlearn(trainX, trainY, testX, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainX, trainY)
    predictions_scikitlearn = knn.predict(testX)
    return predictions_scikitlearn