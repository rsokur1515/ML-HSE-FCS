import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    

    sorted_indices = np.argsort(feature_vector)
    feature_vector_sorted = feature_vector[sorted_indices]
    target_vector_sorted = target_vector[sorted_indices]

    unique_thresholds = (np.unique(feature_vector_sorted[1:]) + np.unique(feature_vector_sorted[:-1])) / 2

    cumulative_sum = np.cumsum(target_vector_sorted)
    cumulative_count = np.arange(1, len(target_vector_sorted) + 1)

    split_indices = np.searchsorted(feature_vector_sorted, unique_thresholds)

    p_left = np.take(cumulative_sum, split_indices - 1) / np.take(cumulative_count, split_indices - 1)
    denominator = len(target_vector_sorted) - np.take(cumulative_count, split_indices - 1)
    p_right = np.where(denominator != 0, 
                       (cumulative_sum[-1] - np.take(cumulative_sum, split_indices - 1)) / denominator, 
                       0)

    h_l = 1 - p_left**2 - (1 - p_left)**2
    h_r = 1 - p_right**2 - (1 - p_right)**2

    ginis = -np.take(cumulative_count, split_indices - 1) / len(feature_vector_sorted) * h_l - \
       (len(target_vector_sorted) - np.take(cumulative_count, split_indices - 1)) / len(feature_vector_sorted) * h_r
    ginis = np.where(np.isnan(ginis), -np.inf, ginis)
    
    index_best = np.argmax(ginis)
    threshold_best = unique_thresholds[index_best]
    gini_best = ginis[index_best]

    threshold_best, gini_best
    
    thresholds = unique_thresholds 

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]): # тут была ошибка №1
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]): # еще ошибка (уже устал их считать) - был 1
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count #тут ошибка
                    
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                

                feature_vector = np.fromiter(map(lambda x: categories_map[x], sub_X[:, feature]), dtype=int) # ошибка №2
            else:
                print(f"Error in _fit_node: feature_best={feature_best}, split={split}")
                raise ValueError

            if len(np.unique(feature_vector)) == 1: # ошибка №3
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "сategorical": # ошибка №5
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    print(feature_type)
                    raise ValueError
                    
             

        if (feature_best is None) or \
   (self._min_samples_leaf is not None and (len(sub_y[split]) < self._min_samples_leaf or len(sub_y[~split]) < self._min_samples_leaf)) or \
   (self._max_depth is not None and depth >= self._max_depth):
 # ошибка № какая-то - не было остальных критерий останов
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #ошибка №4
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            print('Тут ошибка?')
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1) # и тут была ошибка
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1) # ошибка №6

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        threshold = node["threshold"] if self._feature_types[feature] == "real" else node["categories_split"]

        if self._feature_types[feature] == 'real':
            return self._predict_node(x, node["left_child"]) if x[feature] < threshold else self._predict_node(x, node["right_child"])
        else:
            return self._predict_node(x, node["left_child"]) if x[feature] in threshold else self._predict_node(x, node["right_child"])


    
    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
