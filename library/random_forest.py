from concurrent.futures import process
import numpy as np
from joblib import Parallel, delayed
from numpy.random.mtrand import rand, random
from seaborn.utils import desaturate
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import sys
import pandas as pd
from datetime import datetime
from library.utility import confussion
# sys.setrecursionlimit(1000000)


class Node:
    def __init__(self, predicted_class, predicted_score):
        self.predicted_class = predicted_class
        self.predicted_score = predicted_score
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth = 100, min_samples = 2, max_features = None, random_state = None):
        self.max_depth = int(max_depth)
        self.min_samples = int(min_samples)
        self.max_features = int(max_features)
        if random_state :
            self.random_state = int(random_state)
        else :
            self.random_state = None
        self.feat_idxs = None
        self.count = 0

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.max_features = X.shape[1] if not self.max_features else min(self.max_features, X.shape[1])
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.asarray([self._predict(inputs) for inputs in X])

    def predict_score(self, X):
        return np.asarray([self._predict_score(inputs) for inputs in X])

    def _best_split(self, X, y, feat_idxs):
        n_samples, n_features = X.shape

        if n_samples <= 1 :
            return None, None
        
        n_cls_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = self._gini(n_cls_parent)
        best_idx, best_thr = None, None

        # print('idx = ', feat_idxs, 'state = ', self.random_state)
        for idx in feat_idxs : #
            thresholds, labels = zip(*sorted(zip(X[:, idx], y)))
            n_cls_left = [0] * self.n_classes #label kiri awalnya 0 -> ditambah satu persatu
            n_cls_right = n_cls_parent.copy() #label kanan awalnya = parent -> dikurangi satu persatu

            for n in range(1, n_samples):
                label = labels[n - 1] #label
                n_cls_left[label] += 1 #jika label = 0 -> n label kiri 0 ditambah, sama untuk label 1
                n_cls_right[label] -= 1 #jika label = 0 -> n label kanan 0 dikurangi, sama untuk label 1

                gini_left = self._gini(n_cls_left)
                gini_right = self._gini(n_cls_right)
                gini = (n * gini_left + (n_samples - n) * gini_right) / n_samples

                if thresholds[n] == thresholds[n - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[n] + thresholds[n - 1]) / 2
      
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_samples_class = np.asarray([np.sum(y == i) for i in range(self.n_classes)], dtype='int64') # [kelas0, kelas1]
        predicted_class = np.argmax(n_samples_class) #kelas mayoritas = kelas prediksi
        # print(n_samples_class)
        if len(y) == 0 :
            pred_score = np.asarray([1, 0])
        else :
            pred_score = n_samples_class/n_samples

        node = Node(predicted_class=predicted_class, predicted_score=pred_score)

        # print(self.count, end = ' ')
        self.count = self.count + self.min_samples + 1

        if self.random_state :
            feat_idxs = np.random.RandomState(self.random_state + self.count).choice(n_features, self.max_features, replace=False)
        else :
            feat_idxs = np.random.choice(n_features, self.max_features, replace = False)

        if depth < self.max_depth or n_samples > self.min_samples : #self.n_samples
            # print(depth, self.max_depth, n_samples, self.min_samples)
            idx, thr = self._best_split(X, y, feat_idxs)

            if idx is not None :
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _gini(self, n_class) :
        n_class = np.asarray(n_class)
        pct = n_class/sum(n_class) 
        gini = 1 - sum(p**2 for p in pct)
        return gini

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class

    def _predict_score(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_score

class RandomForest:
    def __init__(self, n_trees = 10, min_samples = 2, max_depth = 10, max_features = None, random_state = None, progress = True):
        self.n_trees = n_trees
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.progress = progress
        self.cm = confussion()
        self.trees = []
        self.X_oob = []
        self.y_oob = []
        self.y_oob_pred = []

    def fit(self, X, y):
        print('\nRandom Forest Training')
        self.n_classes = len(set(y))
        self.X = X
        self.y = y
        pool = Pool(processes=4)
        tasks = range(self.n_trees)
        decision_out = []

        if self.progress :
            decision_out = Parallel(n_jobs =4,verbose = 0, backend='loky')(delayed(self._decision)(X, y, i) for i in tqdm(range(self.n_trees)))
            # for i in tqdm(pool.imap_unordered(self._decision, tasks), total = len(tasks)) :
            #     decision_out.append(i)
            # pool.close()
            # pool.join()
        else :
            decision_out = Parallel(n_jobs = 4,verbose=0, backend='loky')(delayed(self._decision)(X, y, i) for i in range(self.n_trees))
            # decision_out = pool.map(self._decision, tasks)
            # pool.close()
            # pool.join()
        
        for i in decision_out :
            self.trees.append(i[0])
            self.X_oob.append(i[1])
            self.y_oob.append(i[2])
            self.y_oob_pred.append(i[3])

    def _decision(self,X, y, i):
        if self.random_state :
            tree = DecisionTree(min_samples = self.min_samples, max_depth = self.max_depth, max_features=self.max_features, random_state=self.random_state + i)
        else :
            tree = DecisionTree(min_samples = self.min_samples, max_depth = self.max_depth, max_features=self.max_features)
        X_random, y_random, X_oob, y_oob = self._random_data(X, y, i)
        tree.fit(X_random, y_random)
        y_oob_pred = tree.predict(X_oob)
        return tree, X_oob, np.asarray(y_oob), np.asarray(y_oob_pred)

    def predict(self, X):
        pred = np.asarray([tree.predict(X) for tree in self.trees])
        pred = np.swapaxes(pred, 0, 1)
        y_pred = []
        for i in pred :
            n_per_class = [np.sum(i == c) for c in range(self.n_classes)]
            pred = np.argmax(n_per_class)
            y_pred.append(pred)
        
        return np.asarray(y_pred)

    def predict_score(self, X):
        pred = np.asarray([tree.predict_score(X) for tree in self.trees])
        pred = np.swapaxes(pred, 0, 1)

        y_pred_score = []
        for i in pred :
            pred = np.mean(i, axis = 0)
            y_pred_score.append(pred)
    
        return np.asarray(y_pred_score)

    def _random_data(self, X, y, i):
        n_samples, n_features = X.shape

        if self.random_state :
            idxs = np.random.RandomState(self.random_state + i).choice(n_samples, n_samples)
        else :
            idxs = np.random.choice(n_samples, n_samples)

        X_bootstrap = X[idxs]
        y_bootstrap = y[idxs]

        # print(feat_idxs)

        oob_idxs = np.asarray([i for i in range(n_samples) if i not in idxs])

        X_oob = X_bootstrap[oob_idxs]
        y_oob = y_bootstrap[oob_idxs]

        return X_bootstrap, y_bootstrap, X_oob, y_oob

    def cur_time(self):
        return f'{datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}' 

    def oob_score(self):
        accuracy = []
        precision = []
        recall = []
        f1_score = []

        for i in range(len(self.y_oob)) :
            y_test = np.asarray(self.y_oob[i])
            y_pred = np.asarray(self.y_oob_pred[i])

            _, _, acc, prec, rec, f1 = self.cm.matrix(y_real = y_test, y_pred = y_pred, metric=True, plot=False)

            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)
            f1_score.append(f1)
            
        accuracy = np.mean(np.asarray(accuracy))
        precision = np.mean(np.asarray(precision))
        recall = np.mean(np.asarray(recall))
        f1_score = np.mean(np.asarray(f1_score))

        return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    import sys
    from sklearn.datasets import load_iris

    dataset = load_iris()
    X, y = dataset.data, dataset.target  # pylint: disable=no-member
    clf = RandomForest(n_trees=10, max_depth=5, min_samples=10, max_features=1, random_state=42)
    clf.fit(X, y)
    # print(clf.predict([[0, 0, 5, 1.5], [0, 0, 1, 1], [1, 5, 1, 0]]))
    # print(clf.predict_score([[0, 0, 5, 1.5], [0, 0, 1, 1], [1, 5, 1, 0]]))
    print(clf.oob_score())