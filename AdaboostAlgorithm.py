import numpy as np
import pandas as pd
import sys

class DecisionStumpMethod:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def PredictionMethod(self, X):
        noOfSamples = X.shape[0]
        valColumn = X[:, self.feature_idx]
        predictions = np.ones(noOfSamples)
        if self.polarity == 1:
            predictions[valColumn < self.threshold] = -1
        else:
            predictions[valColumn > self.threshold] = -1

        return predictions


class AdaboostAlgorithm:
    def __init__(self, k):
        self.k = k
        self.clfs = []

    def fitModule(self, X, y):
        noOfSamples, noOfFeature = X.shape
        w = np.full(noOfSamples, (1 / noOfSamples))
        self.clfs = []
        for x in range(self.k):
            clf = DecisionStumpMethod()
            calculateMinError = float("inf")

            for i in range(noOfFeature):
                valColumn = X[:, i]
                thresholds = np.unique(valColumn)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(noOfSamples)
                    predictions[valColumn < threshold] = -1

                    misclassifiedValue = w[y != predictions]
                    errorFound = sum(misclassifiedValue)

                    if errorFound > 0.5:
                        errorFound = 1 - errorFound
                        p = -1

                    if errorFound < calculateMinError:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = i
                        calculateMinError = errorFound

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - calculateMinError + EPS) / (calculateMinError + EPS))

            predictions = clf.PredictionMethod(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def PredictionMethod(self, X):
        clf_preds = [clf.alpha * clf.PredictionMethod(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    dataset = sys.argv[1]
    def calculateAccuracy(y_true, y_pred):
        calculateAccuracy = np.sum(y_true == y_pred) / len(y_true)
        return calculateAccuracy

    if dataset == "car":
        data = pd.read_csv('car.data', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])
        data["decision"].replace(["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace = True)
        data["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace = True)
        data["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace = True)
        data["persons"].replace(["more"], [4], inplace = True)
        data["doors"].replace(["5more"], [6], inplace = True)
        data["maint"].replace(["vhigh", "high", "med", "low"], [4, 3, 2, 1], inplace = True)
        data["buying"].replace(["vhigh", "high", "med", "low"], [4, 3, 2, 1], inplace = True)

        data['decision'] = data['decision'].astype(int)
        data['safety'] = data['safety'].astype(int)
        data['lug_boot'] = data['lug_boot'].astype(int)
        data['persons'] = data['persons'].astype(int)
        data['doors'] = data['doors'].astype(int)
        data['maint'] = data['maint'].astype(int)
        data['buying'] = data['buying'].astype(int)

        k = 2
        model = AdaboostAlgorithm(k)
        for i in range(0,10):
            data = data.sample(frac=1)
            x, y = data.iloc[:, 0:6].to_numpy(), data.iloc[:,6].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fitModule(X_train, y_train)
                predictedValues = model.PredictionMethod(X_test)
                acc = calculateAccuracy(predictedValues, y_test)
            print("calculateAccuracy:", acc)

    elif dataset == "breast_cancer":
        data = pd.read_csv('breastCancer.data', names=["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension", "Diagnosis"])
        #print (data.dtypes)
        ObjectColumns = data.select_dtypes(include=np.object).columns.tolist()
        data['concavity'] = pd.to_numeric(data['concavity'], errors='coerce')
        data = data.replace(np.nan, data['concavity'].mean(), regex=True)
        data["concavity"] = data["concavity"].astype(int)
        #print(data["concavity"].iloc[23])

        k = 2
        model = AdaboostAlgorithm(k)
        for i in range(0,10):
            data = data.sample(frac=1)
            #x, y = data.iloc[:,1:8].to_numpy(), data.iloc[:,8].to_numpy()
            x, y = data.iloc[:,1:10].to_numpy(), data.iloc[:,10].to_numpy()   #this is giving 0 calculateAccuracy
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fitModule(X_train, y_train)
                predictedValues = model.PredictionMethod(X_test)
                acc = calculateAccuracy(predictedValues, y_test)
            print("calculateAccuracy:", acc)

    elif dataset == "letter":
        data = pd.read_csv('letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
        print (data.dtypes)
        ObjectColumns = data.select_dtypes(include=np.object).columns.tolist()
        data['lettr'] = [ord(item)-64 for item in data['lettr']]
        print(data["lettr"].iloc[23])

        k = 2
        model = AdaboostAlgorithm(k)
        for i in range(0, 10):
            data = data.sample(frac=1)
            x, y = data.iloc[:, 1:17].to_numpy(), data.iloc[:, 0].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fitModule(X_train, y_train)
                predictedValues = model.PredictionMethod(X_test)
                acc = calculateAccuracy(predictedValues, y_test)
            print("calculateAccuracy:", acc)

    elif dataset == "mushroom":
        data = pd.read_csv('mushroom.data', names=["decision", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment", 
        "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", 
        "population", "habitat"])

        data["decision"].replace(["e", "p"], [0, 1], inplace = True)
        data["cap-shape"].replace(["b", "c", "x", "f", "k", "s"], [0, 1, 2, 3, 4, 5], inplace = True)
        data["cap-surface"].replace(["f", "g", "y", "s"], [0, 1, 2, 3], inplace = True)
        data["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
        data["bruises"].replace(["t", "f"], [0, 1], inplace = True)
        data["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
        data["gill-attachment"].replace(["a", "d", "f", "n"], [0, 1, 2, 3], inplace = True)
        data["gill-spacing"].replace(["c", "w", "d"], [0, 1, 2], inplace = True)
        data["gill-size"].replace(["b", "n"], [0, 1], inplace = True)
        data["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace = True)
        data["stalk-shape"].replace(["e", "t"], [0, 1], inplace = True)
        data["stalk-root"].replace(["b", "c", "u", "e", "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace = True)
        data["stalk-surface-above-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace = True)
        data["stalk-surface-below-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace = True)
        data["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [1,2,3,4,5,6,7,8,9], inplace = True)
        data["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"], [1,2,3,4,5,6,7,8,9], inplace = True)
        data["veil-type"].replace(["p", "u"], [1, 2], inplace = True)
        data["veil-color"].replace(["n", "o", "w", "y"], [1, 2, 3, 4], inplace = True)
        data["ring-number"].replace(["n", "o", "t"], [1, 2, 3], inplace = True)
        data["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [1, 2, 3, 4, 5, 6, 7, 8], inplace = True)
        data["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [1,2,3,4,5,6,7,8,9], inplace = True)
        data["population"].replace(["a", "c", "n", "s", "v", "y"], [1, 2, 3, 4, 5, 6], inplace = True)
        data["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [1, 2, 3, 4, 5, 6, 7], inplace = True)

        data["  1"] = data['decision'].astype(int)
        data["cap-shape"] = data['cap-shape'].astype(int)
        data["cap-surface"] = data['cap-surface'].astype(int)
        data["cap-color"] = data['cap-color'].astype(int)
        data["bruises"] = data['bruises'].astype(int)
        data["odor"] = data['odor'].astype(int)
        data["gill-attachment"] = data['gill-attachment'].astype(int)
        data["gill-spacing"] = data['gill-spacing'].astype(int)
        data["gill-size"] = data['gill-size'].astype(int)
        data["gill-color"] = data['gill-color'].astype(int)
        data["stalk-shape"] = data['stalk-shape'].astype(int)
        data["stalk-root"] = data['stalk-root'].astype(int)
        data["stalk-surface-above-ring"] = data['stalk-surface-above-ring'].astype(int)
        data["stalk-surface-below-ring"] = data['stalk-surface-below-ring'].astype(int)
        data["stalk-color-above-ring"] = data['stalk-color-above-ring'].astype(int)
        data["stalk-color-below-ring"] = data['stalk-color-below-ring'].astype(int)
        data["veil-type"] = data['veil-type'].astype(int)
        data["veil-color"] = data['veil-color'].astype(int)
        data["ring-number"] = data['ring-number'].astype(int)
        data["ring-type"] = data['ring-type'].astype(int)
        data["spore-print-color"] = data['spore-print-color'].astype(int)
        data["population"] = data['population'].astype(int)
        data["habitat"] = data['habitat'].astype(int)

        k = 2
        model = AdaboostAlgorithm(k)
        for i in range(1, 11):
            data = data.sample(frac=1)
            x, y = data.iloc[:, 1:23].to_numpy(), data.iloc[:, 0].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fitModule(X_train, y_train)
                predictedValues = model.PredictionMethod(X_test)
                acc = calculateAccuracy(predictedValues, y_test)
            print("calculateAccuracy:",i, acc)

print("Entire out")