from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, test):
        predictions = []

        for row in test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, point):
        best_dist = distance.euclidean(point, self.x_train[0])
        best_index = 0

        for index in range(1, len(self.x_train)):
            dist = distance.euclidean(point, self.x_train[index])

            if dist < best_dist:
                best_dist = dist
                best_index = index

        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .5)

classifier = ScrappyKNN()

classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)

print "Prediction Accuracy: ", accuracy_score(y_test, predictions)
